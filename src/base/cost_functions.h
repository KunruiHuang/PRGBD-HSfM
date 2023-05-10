// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_BASE_COST_FUNCTIONS_H_
#define COLMAP_SRC_BASE_COST_FUNCTIONS_H_

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

// Standard bundle adjustment cost function for variable
// camera pose and calibration and point parameters.
template <typename CameraModel>
class BundleAdjustmentCostFunction {
 public:
  explicit BundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            BundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 3,
            CameraModel::kNumParams>(
        new BundleAdjustmentCostFunction(point2D)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    // Rotate and translate.
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection[0] += tvec[0];
    projection[1] += tvec[1];
    projection[2] += tvec[2];

    // Project to image plane.
    projection[0] /= projection[2];
    projection[1] /= projection[2];

    // Distort and transform to pixel space.
    CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                              &residuals[0], &residuals[1]);

    // Re-projection error.
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);

    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
};

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.
template <typename CameraModel>
class BundleAdjustmentConstantPoseCostFunction {
 public:
  BundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d& qvec,
                                           const Eigen::Vector3d& tvec,
                                           const Eigen::Vector2d& point2D)
      : qw_(qvec(0)),
        qx_(qvec(1)),
        qy_(qvec(2)),
        qz_(qvec(3)),
        tx_(tvec(0)),
        ty_(tvec(1)),
        tz_(tvec(2)),
        observed_x_(point2D(0)),
        observed_y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
                                     const Eigen::Vector3d& tvec,
                                     const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            BundleAdjustmentConstantPoseCostFunction<CameraModel>, 2, 3,
            CameraModel::kNumParams>(
        new BundleAdjustmentConstantPoseCostFunction(qvec, tvec, point2D)));
  }

  template <typename T>
  bool operator()(const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

    // Rotate and translate.
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection[0] += T(tx_);
    projection[1] += T(ty_);
    projection[2] += T(tz_);

    // Project to image plane.
    projection[0] /= projection[2];
    projection[1] /= projection[2];

    // Distort and transform to pixel space.
    CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                              &residuals[0], &residuals[1]);

    // Re-projection error.
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);

    return true;
  }

 private:
  const double qw_;
  const double qx_;
  const double qy_;
  const double qz_;
  const double tx_;
  const double ty_;
  const double tz_;
  const double observed_x_;
  const double observed_y_;
};

// Rig bundle adjustment cost function for variable camera pose and calibration
// and point parameters. Different from the standard bundle adjustment function,
// this cost function is suitable for camera rigs with consistent relative poses
// of the cameras within the rig. The cost function first projects points into
// the local system of the camera rig and then into the local system of the
// camera within the rig.
template <typename CameraModel>
class RigBundleAdjustmentCostFunction {
 public:
  explicit RigBundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            RigBundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 4, 3, 3,
            CameraModel::kNumParams>(
        new RigBundleAdjustmentCostFunction(point2D)));
  }

  template <typename T>
  bool operator()(const T* const rig_qvec, const T* const rig_tvec,
                  const T* const rel_qvec, const T* const rel_tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    // Concatenate rotations.
    T qvec[4];
    ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

    // Concatenate translations.
    T tvec[3];
    ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
    tvec[0] += rel_tvec[0];
    tvec[1] += rel_tvec[1];
    tvec[2] += rel_tvec[2];

    // Rotate and translate.
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection[0] += tvec[0];
    projection[1] += tvec[1];
    projection[2] += tvec[2];

    // Project to image plane.
    projection[0] /= projection[2];
    projection[1] /= projection[2];

    // Distort and transform to pixel space.
    CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                              &residuals[0], &residuals[1]);

    // Re-projection error.
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);

    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
};

// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `SphereManifold`.
class RelativePoseCostFunction {
 public:
  RelativePoseCostFunction(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
      : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& x1,
                                     const Eigen::Vector2d& x2) {
    return (new ceres::AutoDiffCostFunction<RelativePoseCostFunction, 1, 4, 3>(
        new RelativePoseCostFunction(x1, x2)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  T* residuals) const {
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
    ceres::QuaternionToRotation(qvec, R.data());

    // Matrix representation of the cross product t x R.
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -tvec[2], tvec[1], tvec[2], T(0), -tvec[0], -tvec[1], tvec[0],
        T(0);

    // Essential matrix.
    const Eigen::Matrix<T, 3, 3> E = t_x * R;

    // Homogeneous image coordinates.
    const Eigen::Matrix<T, 3, 1> x1_h(T(x1_), T(y1_), T(1));
    const Eigen::Matrix<T, 3, 1> x2_h(T(x2_), T(y2_), T(1));

    // Squared sampson error.
    const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
    const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
    const T x2tEx1 = x2_h.transpose() * Ex1;
    residuals[0] = x2tEx1 * x2tEx1 /
                   (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                    Etx2(1) * Etx2(1));

    return true;
  }

 private:
  const double x1_;
  const double y1_;
  const double x2_;
  const double y2_;
};

// Prior Orientations cost function
class PriorOrientationCostFunction {
 public:
  PriorOrientationCostFunction(const Eigen::Vector4d& prior,
                               const double& sqrt_information)
      : prior_(prior), sqrt_information_(sqrt_information) {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& prior,
                                     const double sqrt_information) {
    return (new ceres::AutoDiffCostFunction<PriorOrientationCostFunction, 3, 4>(
        new PriorOrientationCostFunction(prior, sqrt_information)));
  }

  template <typename T>
  bool operator()(const T* const qvec, T* residuals) const {
    Eigen::Matrix<T, 4, 1> prior_qvec = prior_.cast<T>();
    T delta_q[4];
    ceres::QuaternionProduct(qvec, prior_qvec.data(), delta_q);
    // set the residuals
    residuals[0] = T(sqrt_information_) * T(2.0) * delta_q[1];
    residuals[1] = T(sqrt_information_) * T(2.0) * delta_q[2];
    residuals[2] = T(sqrt_information_) * T(2.0) * delta_q[3];

    return true;
  }

 private:
  const Eigen::Vector4d prior_;
  const double sqrt_information_;
};

class PriorPositionCostFunction {
 public:
  PriorPositionCostFunction(const Eigen::Vector3d& prior,
                            const double sqrt_information)
      : prior_(prior), sqrt_information_(sqrt_information) {}

  static ceres::CostFunction* Create(const Eigen::Vector3d& prior,
                                     const double sqrt_information) {
    return (new ceres::AutoDiffCostFunction<PriorPositionCostFunction, 3, 4, 3>(
        new PriorPositionCostFunction(prior, sqrt_information)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  T* residuals) const {
    Eigen::Matrix<T, 3, 1> prior = prior_.cast<T>();
    // Get the camera center
    T qvec_conjugate[4];
    qvec_conjugate[0] = qvec[0];
    qvec_conjugate[1] = -qvec[1];
    qvec_conjugate[2] = -qvec[2];
    qvec_conjugate[3] = -qvec[3];

    T camera_center[3];
    ceres::UnitQuaternionRotatePoint(qvec_conjugate, tvec, camera_center);
    camera_center[0] *= -1.0;
    camera_center[1] *= -1.0;
    camera_center[2] *= -1.0;

    // Set the residuals
    residuals[0] = T(sqrt_information_) * (prior[0] - camera_center[0]);
    residuals[1] = T(sqrt_information_) * (prior[1] - camera_center[1]);
    residuals[2] = T(sqrt_information_) * (prior[2] - camera_center[2]);

    return true;
  }

 private:
  const Eigen::Vector3d prior_;
  const double sqrt_information_;
};

class PriorDepthCostFunction {
 public:
  PriorDepthCostFunction(const double prior_depth,
                         const double sqrt_information)
      : prior_depth_(prior_depth), sqrt_information_(sqrt_information) {}

  static ceres::CostFunction* Create(const double prior_depth,
                                     const double sqrt_information) {
    return (new ceres::AutoDiffCostFunction<PriorDepthCostFunction, 1, 4, 3, 3>(
        new PriorDepthCostFunction(prior_depth, sqrt_information)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  const T* const point3D, T* residuals) const {
    // Rotate and translate.
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection[0] += tvec[0];
    projection[1] += tvec[1];
    projection[2] += tvec[2];

    residuals[0] = T(sqrt_information_) * (projection[2] - T(prior_depth_));

    return true;
  }

 private:
  const double prior_depth_;
  const double sqrt_information_;
};

class PriorDepthConstantPoseCostFunction {
 public:
  PriorDepthConstantPoseCostFunction(const Eigen::Vector4d& qvec,
                                     const Eigen::Vector3d& tvec,
                                     const double prior_depth,
                                     const double sqrt_information)
      : qw_(qvec(0)),
        qx_(qvec(1)),
        qy_(qvec(2)),
        qz_(qvec(3)),
        tx_(tvec(0)),
        ty_(tvec(1)),
        tz_(tvec(2)),
        prior_depth_(prior_depth),
        sqrt_information_(sqrt_information) {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
                                     const Eigen::Vector3d& tvec,
                                     const double prior_depth,
                                     const double sqrt_information) {
    return (new ceres::AutoDiffCostFunction<PriorDepthConstantPoseCostFunction,
                                            1, 3>(
        new PriorDepthConstantPoseCostFunction(qvec, tvec, prior_depth,
                                               sqrt_information)));
  }

  template <typename T>
  bool operator()(const T* const point3D, T* residuals) const {
    const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

    // Rotate and translate.
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection[0] += T(tx_);
    projection[1] += T(ty_);
    projection[2] += T(tz_);

    residuals[0] = T(sqrt_information_) * (projection[2] - prior_depth_);

    return true;
  }

 private:
  const double qw_;
  const double qx_;
  const double qy_;
  const double qz_;
  const double tx_;
  const double ty_;
  const double tz_;
  const double prior_depth_;
  const double sqrt_information_;
};

class RelativeRotationCostFunction {
 public:
  RelativeRotationCostFunction(const Eigen::Vector3d& relative_rotation,
                               const double sqrt_information)
      : relative_rotation_(relative_rotation),
        sqrt_information_(sqrt_information){};

  static ceres::CostFunction* Create(const Eigen::Vector3d& relative_rotation,
                                     const double sqrt_information) {
    return (
        new ceres::AutoDiffCostFunction<RelativeRotationCostFunction, 3, 3, 3>(
            new RelativeRotationCostFunction(relative_rotation,
                                             sqrt_information)));
  }

  // Check the error by the rotation cycle error rc2w * rc1w^-1 * rc2c1
  template <typename T>
  bool operator()(const T* const angleAxis1, const T* const angleAxis2,
                  T* residuals) const {
    const T relative_rotation[3] = {T(relative_rotation_[0]),
                                    T(relative_rotation_[1]),
                                    T(relative_rotation_[2])};

    Eigen::Matrix<T, 3, 3> Rji, Ri, Rj;
    ceres::AngleAxisToRotationMatrix(relative_rotation, Rji.data());
    ceres::AngleAxisToRotationMatrix(angleAxis1, Ri.data());
    ceres::AngleAxisToRotationMatrix(angleAxis2, Rj.data());

    const Eigen::Matrix<T, 3, 3> cycle_rotation_mat =
        (Rj * Ri.transpose()) * Rji.transpose();
    Eigen::Matrix<T, 3, 1> cycle_rotation;
    ceres::RotationMatrixToAngleAxis(cycle_rotation_mat.data(),
                                     cycle_rotation.data());

    residuals[0] = T(sqrt_information_) * cycle_rotation(0);
    residuals[1] = T(sqrt_information_) * cycle_rotation(1);
    residuals[2] = T(sqrt_information_) * cycle_rotation(2);

    return true;
  }

 private:
  // Rji
  const Eigen::Vector3d relative_rotation_;
  const double sqrt_information_;
};

// global rotations cost function
class GlboalRotationCostFunction {
 public:
  GlboalRotationCostFunction(const Eigen::Vector3d& global_orientation_i,
                             const Eigen::Vector3d& global_orientation_j,
                             const double sqrt_information)
      : global_orientation_i_(global_orientation_i),
        global_orientation_j_(global_orientation_j),
        sqrt_information_(sqrt_information) {}

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& global_orientation_i,
      const Eigen::Vector3d& global_orientation_j,
      const double sqrt_information) {
    return (
        new ceres::AutoDiffCostFunction<GlboalRotationCostFunction, 3, 4, 4>(
            new GlboalRotationCostFunction(
                global_orientation_i, global_orientation_j, sqrt_information)));
  }

  template <typename T>
  bool operator()(const T* const qvec_i, const T* const qvec_j,
                  T* residuals) const {
    const T gri[3] = {T(global_orientation_i_[0]), T(global_orientation_i_[1]),
                      T(global_orientation_i_[2])};
    const T grj[3] = {T(global_orientation_j_[0]), T(global_orientation_j_[1]),
                      T(global_orientation_j_[2])};

    T GQi[4], GQj[4];
    ceres::AngleAxisToQuaternion(gri, GQi);
    ceres::AngleAxisToQuaternion(grj, GQj);

    T GQi_con[4];
    GQi_con[0] = GQi[0];
    GQi_con[1] = -GQi[1];
    GQi_con[2] = -GQi[2];
    GQi_con[3] = -GQi[3];

    T Qj_con[4]; 
    Qj_con[0] = qvec_j[0]; 
    Qj_con[1] = -qvec_j[1];
    Qj_con[2] = -qvec_j[2];
    Qj_con[3] = -qvec_j[3];

    T GQji[4], Qij[4], deltaQ[4];
    ceres::QuaternionProduct(GQj, GQi_con, GQji); 
    ceres::QuaternionProduct(qvec_i, Qj_con, Qij); 
    ceres::QuaternionProduct(GQji, Qij, deltaQ);

     // set the residuals
    residuals[0] = T(sqrt_information_) * T(2.0) * deltaQ[1];
    residuals[1] = T(sqrt_information_) * T(2.0) * deltaQ[2];
    residuals[2] = T(sqrt_information_) * T(2.0) * deltaQ[3];

    return true;
  }

 private:
  // global rotation measurement
  const Eigen::Vector3d global_orientation_i_;
  const Eigen::Vector3d global_orientation_j_;
  const double sqrt_information_;
};

inline void SetQuaternionManifold(ceres::Problem* problem, double* qvec) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
  problem->SetManifold(qvec, new ceres::QuaternionManifold);
#else
  problem->SetParameterization(qvec, new ceres::QuaternionParameterization);
#endif
}

inline void SetSubsetManifold(int size, const std::vector<int>& constant_params,
                              ceres::Problem* problem, double* params) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
  problem->SetManifold(params,
                       new ceres::SubsetManifold(size, constant_params));
#else
  problem->SetParameterization(
      params, new ceres::SubsetParameterization(size, constant_params));
#endif
}

template <int size>
inline void SetSphereManifold(ceres::Problem* problem, double* params) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
  problem->SetManifold(params, new ceres::SphereManifold<size>);
#else
  problem->SetParameterization(
      params, new ceres::HomogeneousVectorParameterization(size));
#endif
}

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_COST_FUNCTIONS_H_
