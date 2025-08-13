import numpy as np

class KalmanFilter:
    """
    ByteTrack 호환 Kalman Filter (급격한 방향 변화 대응 버전)
    상태: [cx, cy, a, h, vx, vy, va, vh]
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # 상태 전이 행렬(F)
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # 관측 행렬(H)
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 기본 표준편차 (ByteTrack 기본값보다 관측 노이즈 살짝 높임)
        self.std_weight_position = 1.0 / 20
        self.std_weight_velocity = 1.0 / 160

        # Adaptive Q 계수
        self.adaptive_q_coeff = 0.8
        self.max_adaptive_scale = 15.0

    def initiate(self, measurement):
        """새 트랙 초기화"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self.std_weight_position * measurement[3],  # cx
            2 * self.std_weight_position * measurement[3],  # cy
            1e-2,                                           # a
            2 * self.std_weight_position * measurement[3],  # h
            10 * self.std_weight_velocity * measurement[3], # vx (크게 줌)
            10 * self.std_weight_velocity * measurement[3], # vy
            1e-5,                                           # va
            10 * self.std_weight_velocity * measurement[3]  # vh
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """단일 트랙 예측"""
        std_pos = self.std_weight_position * mean[3]
        std_vel = self.std_weight_velocity * mean[3]

        motion_cov = np.diag([
            std_pos**2, std_pos**2, 1e-4, std_pos**2,
            std_vel**2, std_vel**2, 1e-6, std_vel**2
        ])

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov
        return mean, covariance

    def multi_predict(self, mean, covariance):
        """여러 트랙 동시 예측"""
        std_pos = self.std_weight_position * mean[:, 3]
        std_vel = self.std_weight_velocity * mean[:, 3]

        motion_cov = np.stack([
            np.diag([
                sp**2, sp**2, 1e-4, sp**2,
                sv**2, sv**2, 1e-6, sv**2
            ])
            for sp, sv in zip(std_pos, std_vel)
        ])

        mean = np.dot(mean, self._motion_mat.T)
        for i in range(mean.shape[0]):
            covariance[i] = (
                np.linalg.multi_dot((self._motion_mat, covariance[i], self._motion_mat.T))
                + motion_cov[i]
            )
        return mean, covariance

    def project(self, mean, covariance):
        """상태 -> 관측 공간 투영"""
        std = [
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-1,
            self.std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        ))
        return mean, covariance + innovation_cov

    def multi_project(self, mean, covariance):
        """여러 트랙 동시에 관측 공간으로 투영"""
        std = [
            self.std_weight_position * mean[:, 3],
            self.std_weight_position * mean[:, 3],
            np.full_like(mean[:, 3], 1e-1),
            self.std_weight_position * mean[:, 3]
        ]
        innovation_cov = np.stack([
            np.diag(np.square(s)) for s in zip(*std)
        ])

        mean = np.dot(mean, self._update_mat.T)
        for i in range(mean.shape[0]):
            covariance[i] = (
                np.linalg.multi_dot((self._update_mat, covariance[i], self._update_mat.T))
                + innovation_cov[i]
            )
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """관측값 업데이트 (Adaptive Q 적용)"""
        projected_mean, projected_cov = self.project(mean, covariance)

        # Innovation
        innovation = measurement - projected_mean
        try:
            chol_factor = np.linalg.cholesky(projected_cov)
            lower = True
        except np.linalg.LinAlgError:
            chol_factor = np.linalg.cholesky(projected_cov + 1e-6 * np.eye(len(projected_cov)))
            lower = True

        kalman_gain = np.linalg.solve(
            chol_factor.T,
            np.linalg.solve(chol_factor, np.dot(covariance, self._update_mat.T).T)
        ).T

        # Adaptive Q 스케일 계산
        innov_norm = np.sqrt(np.dot(innovation.T, np.linalg.solve(projected_cov, innovation)))
        q_scale = min(1.0 + self.adaptive_q_coeff * innov_norm, self.max_adaptive_scale)

        mean = mean + np.dot(kalman_gain, innovation)
        covariance = covariance - np.dot(kalman_gain, np.dot(projected_cov, kalman_gain.T))

        # 속도 관련 공분산 확대 (급격한 변화 대응)
        for i in range(4, 8):
            covariance[i, i] *= q_scale

        return mean, covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """게이팅 거리 계산"""
        projected_mean, projected_cov = self.project(mean, covariance)
        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean
        z = np.linalg.solve(cholesky_factor, d.T)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
