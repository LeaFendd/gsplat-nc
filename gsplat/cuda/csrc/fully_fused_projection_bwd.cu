#include "bindings.h"
#include "helpers.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

/****************************************************************************
 * Projection of Gaussians (Single Batch) Backward Pass
 ****************************************************************************/

template <typename T>
__global__ void fully_fused_projection_bwd_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const vec3<T> *__restrict__ means,  // [N, 3]
    const vec4<T> *__restrict__ quats,  // [N, 4]
    const vec3<T> *__restrict__ scales, // [N, 3]
    const T *__restrict__ viewmats,     // [C, 4, 4]
    const T *__restrict__ Ks,           // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const T eps2d,
    // fwd outputs
    const int32_t *__restrict__ radii,   // [C, N]
    const vec3<T> *__restrict__ conics,  // [C, N, 3]
    const T *__restrict__ compensations, // [C, N] optional
    // grad outputs
    const vec2<T> *__restrict__ v_means2d, // [C, N, 2]
    const T *__restrict__ v_depths,        // [C, N]
    const vec3<T> *__restrict__ v_conics,  // [C, N, 3]
    const vec3<T> *__restrict__ v_normals, // [C, N, 3]
    const T *__restrict__ v_compensations, // [C, N] optional
    // grad inputs
    vec3<T> *__restrict__ v_means,  // [N, 3]
    vec4<T> *__restrict__ v_quats,  // [N, 4]
    vec3<T> *__restrict__ v_scales, // [N, 3]
    T *__restrict__ v_viewmats      // [C, 4, 4] optional
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx] <= 0) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    vec3<T> mean = means[gid];
    vec4<T> quat = quats[gid];
    vec3<T> scale = scales[gid];

    viewmats += cid * 16;
    Ks += cid * 9;

    vec3<T> conic = conics[idx];

    vec2<T> v_mean2d = v_means2d[idx];
    T v_depth = v_depths[idx];
    vec3<T> v_conic = v_conics[idx];
    vec3<T> v_normal = v_normals[idx];

    // vjp: compute the inverse of the 2d covariance
    mat2<T> covar2d_inv = mat2<T>(conic[0], conic[1], conic[1], conic[2]);
    mat2<T> v_covar2d_inv =
        mat2<T>(v_conic[0], v_conic[1] * .5f, v_conic[1] * .5f, v_conic[2]);
    mat2<T> v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const T compensation = compensations[idx];
        const T v_compensation = v_compensations[idx];
        add_blur_vjp(
            eps2d, covar2d_inv, compensation, v_compensation, v_covar2d
        );
    }

    // transform Gaussian to camera space
    mat3<T> R = mat3<T>(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

    mat3<T> covar;

    // compute from quaternions and scales

    quat_scale_to_covar_preci<T>(quat, scale, &covar, nullptr);

    vec3<T> mean_c;
    pos_world_to_cam(R, t, mean, mean_c);
    mat3<T> covar_c;
    covar_world_to_cam(R, covar, covar_c);

    // vjp: perspective projection
    T fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3<T> v_covar_c(0.f);
    vec3<T> v_mean_c(0.f);
    persp_proj_vjp<T>(
        mean_c,
        covar_c,
        fx,
        fy,
        cx,
        cy,
        image_width,
        image_height,
        v_covar2d,
        v_mean2d,
        v_mean_c,
        v_covar_c
    );

    // add contribution from v_depths
    v_mean_c.z += v_depth;

    // vjp: transform Gaussian covariance to camera space
    vec3<T> v_mean(0.f);
    mat3<T> v_covar(0.f);
    mat3<T> v_R(0.f);
    vec3<T> v_t(0.f);
    pos_world_to_cam_vjp(R, t, mean, v_mean_c, v_R, v_t, v_mean);
    covar_world_to_cam_vjp(R, covar, v_covar_c, v_R, v_covar);

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid;
            gpuAtomicAdd(&(v_means->x), v_mean.x);
            gpuAtomicAdd(&(v_means->y), v_mean.y);
            gpuAtomicAdd(&(v_means->z), v_mean.z);
        }
    }

    // Directly output gradients w.r.t. the quaternion and scale
    mat3<T> rotmat = quat_to_rotmat<T>(quat);
    vec4<T> v_quat(0.f);
    vec3<T> v_scale(0.f);
    quat_scale_to_covar_vjp<T>(quat, scale, rotmat, v_covar, v_quat, v_scale);
    // from v_normal
    v_normal = glm::transpose(R) * v_normal; // to world
    quat_to_normal_vjp<T>(quat, v_normal, v_quat);

    warpSum(v_quat, warp_group_g);
    warpSum(v_scale, warp_group_g);
    if (warp_group_g.thread_rank() == 0) {
        v_quats += gid;
        v_scales += gid;
        gpuAtomicAdd(&(v_quats->x), v_quat.x);
        gpuAtomicAdd(&(v_quats->y), v_quat.y);
        gpuAtomicAdd(&(v_quats->z), v_quat.z);
        gpuAtomicAdd(&(v_quats->w), v_quat.w);
        gpuAtomicAdd(&(v_scales->x), v_scale.x);
        gpuAtomicAdd(&(v_scales->y), v_scale.y);
        gpuAtomicAdd(&(v_scales->z), v_scale.z);
    }

    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 4]
    const torch::Tensor &scales,   // [N, 3]
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    // fwd outputs
    const torch::Tensor &radii,                       // [C, N]
    const torch::Tensor &conics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &compensations, // [C, N] optional
    // grad outputs
    const torch::Tensor &v_means2d,                     // [C, N, 2]
    const torch::Tensor &v_depths,                      // [C, N]
    const torch::Tensor &v_conics,                      // [C, N, 3]
    const torch::Tensor &v_normals,                     // [C, N, 3]
    const at::optional<torch::Tensor> &v_compensations, // [C, N] optional
    const bool viewmats_requires_grad
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);
    CHECK_INPUT(v_normals);
    if (compensations.has_value()) {
        CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_quats = torch::zeros_like(quats);
    torch::Tensor v_scales = torch::zeros_like(scales);

    torch::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros_like(viewmats);
    }
    if (C && N) {
        fully_fused_projection_bwd_kernel<float>
            <<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0, stream>>>(
                C,
                N,
                reinterpret_cast<vec3<float> *>(means.data_ptr<float>()),
                reinterpret_cast<vec4<float> *>(quats.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(scales.data_ptr<float>()),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                eps2d,
                radii.data_ptr<int32_t>(),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                compensations.has_value()
                    ? compensations.value().data_ptr<float>()
                    : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                v_depths.data_ptr<float>(),
                reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(v_normals.data_ptr<float>()),
                v_compensations.has_value()
                    ? v_compensations.value().data_ptr<float>()
                    : nullptr,
                reinterpret_cast<vec3<float> *>(v_means.data_ptr<float>()),
                reinterpret_cast<vec4<float> *>(v_quats.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(v_scales.data_ptr<float>()),
                viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
            );
    }
    return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
}
