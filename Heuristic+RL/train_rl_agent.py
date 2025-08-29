#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PROGRAM NAME  : train_rl_agent.py
PROGRAMMER    : YONG
CREATION DATE : 2025-08-29

[Description]
  - Stable-Baselines3 기반 PPO 강화학습 에이전트 학습 스크립트
  - SchedulingEnv(Gym 환경)을 사용하여 생산 스케줄링 최적화 수행
  - 학습 목표: 주문 납기 준수, 생산 효율, 스타일 연속성, 라인 균등화
  - CustomEvalCallback 사용:
      1. 평가 시 주문 지연(missed_deliveries) 기준으로 성능 체크
      2. 지연 최소화 및 평균 reward 기반 최적 모델 저장
      3. 개선 없을 경우 조기 종료(StopTraining) 가능
  - 학습 환경/평가 환경 모두 SchedulingEnv를 사용하며, Monitor로 info dict 전달
  - 모델/로그 저장 경로 지정 가능
  - 학습 도중 오류 발생 시, 오류 시점 모델을 별도 저장
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from scheduling_env import SchedulingEnv
from stable_baselines3.common.monitor import Monitor

# ========================================
# Custom Evaluation Callback
# ========================================
class CustomEvalCallback(EvalCallback):
    """
    Stable-Baselines3 EvalCallback 확장:
        - 평가 시 주문 지연 수(missed_deliveries) 수집
        - 지연 최소화 기준으로 최적 모델 저장
        - patience/min_evals 등 조기 종료 옵션 포함
    """
    def __init__(self, *args, **kwargs):
        # Pop custom arguments before passing to the parent
        patience = kwargs.pop('patience', 100)    # 개선 없을 경우 최대 대기 step
        min_evals = kwargs.pop('min_evals', 1000) # 최소 평가 횟수
        
        # 부모 클래스 초기화
        super(CustomEvalCallback, self).__init__(*args, **kwargs)
        
        # Custom attribute 초기화
        self.patience = patience
        self.min_evals_grace = min_evals
        self.best_late_orders = float('inf')          # 최적 missed_deliveries
        self.best_reward_at_best_late = -float('inf') # 해당 시점 reward
        self.no_improvement_count = 0
        self.eval_infos = [] # episode info dict 저장 리스트

    def _info_callback(self, local_vars, global_vars):
        """
        evaluate_policy 내부 callback.
        매 episode 종료 시 info dict를 eval_infos에 저장.
        SchedulingEnv에서 반환하는 'missed_deliveries' 확인 가능.
        """
        if local_vars['dones'][0]:
            self.eval_infos.append(local_vars['infos'][0])

    def _on_step(self) -> bool:
         """
        학습 step마다 호출되는 함수.
        - eval_freq마다 evaluate_policy 실행
        - 평균 missed_deliveries 계산
        - 최적 모델 갱신 여부 판단
        - StopTrainingOnNoModelImprovement 호출 가능
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_infos.clear() # 이전 평가 info 초기화

            # 1. evaluate_policy 호출
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._info_callback, # episode info 수집
            )

            # 2. 평균 reward 저장
            mean_reward = np.mean(episode_rewards)
            self.last_mean_reward = mean_reward

            # 3. 평균 missed_deliveries 계산
            if self.eval_infos:
                missed_deliveries_list = [info.get('missed_deliveries', float('inf')) for info in self.eval_infos]
                avg_late_orders = np.mean(missed_deliveries_list)
            else:
                avg_late_orders = float('inf')

            # 4. 새로운 최적 모델 판단
            is_new_best = False
            if avg_late_orders < self.best_late_orders:
                is_new_best = True
            elif avg_late_orders == self.best_late_orders and mean_reward > self.best_reward_at_best_late:
                is_new_best = True

            # 5. 최적 모델 갱신
            if is_new_best:
                self.best_late_orders = avg_late_orders
                self.best_reward_at_best_late = mean_reward
                print(f"CustomEval: New best model found! Avg late orders: {self.best_late_orders:.2f}, Mean reward: {self.best_reward_at_best_late:.2f}")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model.zip"))
                    
            # 6. StopTraining 체크
            if self.callback is not None:
                return self.callback.on_step()

        return True

# ========================================
# 학습 함수
# ========================================
def train_agent(model_path, logs_dir, order_path, info_path, start_date_str, daily_work_hours, train_timesteps, optimized_orders_df=None):
    """
    PPO 에이전트 학습 실행
    Args:
        model_path: 모델 저장 경로
        logs_dir: tensorboard/log 파일 저장 경로
        order_path: 주문 CSV 파일 경로
        info_path: 라인 정보 CSV 파일 경로
        start_date_str: 학습 시작일("YYYY-MM-DD")
        daily_work_hours: 하루 작업 시간
        train_timesteps: 학습 총 timestep 수
        optimized_orders_df: Optional, 사전 최적화 주문 데이터
    """
    # 1. 로그/모델 폴더 생성
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # 2. 기존 모델 존재 여부 확인
    if os.path.exists(f"{model_path}.zip"):
        print("--- 모델이 존재합니다. 새로운 학습을 시작하려면 기존 모델을 삭제해주세요. ---")
    else:
        print("--- 강화학습 에이전트 학습 시작 ---")

        # 3. 학습 환경 생성
        env = SchedulingEnv(order_path=order_path, info_path=info_path, start_date_str=start_date_str, daily_work_hours=daily_work_hours, optimized_orders_df=optimized_orders_df)
        
        # 4. 평가 환경 생성 (Monitor로 info dict 수집)
        eval_env = SchedulingEnv(order_path=order_path, info_path=info_path, start_date_str=start_date_str, daily_work_hours=daily_work_hours, optimized_orders_df=optimized_orders_df)
        eval_env = Monitor(eval_env, filename=os.path.join(logs_dir, "eval_monitor"), info_keywords=('missed_deliveries',))

        # 5. PPO 모델 초기화
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir, device="cpu")

        # 6. Callback 정의
        stop_callback = CustomEvalCallback(eval_env, best_model_save_path=model_path,
                                        log_path=logs_dir, eval_freq=500, # 500 step마다 평가
                                        n_eval_episodes=5, deterministic=True, render=False,
                                        patience=250, min_evals=1000)
        
        eval_callback = CustomEvalCallback(eval_env, best_model_save_path=model_path,
                                        log_path=logs_dir, eval_freq=500, 
                                        n_eval_episodes=5, deterministic=True, render=False,
                                        callback_after_eval=stop_callback)

        # 7. 학습 수행
        print(f"모델 학습 시작: 총 {train_timesteps} timesteps (조기 종료 활성화)\n")
        try:
            model.learn(total_timesteps=train_timesteps, callback=eval_callback)
            print("모델 학습 완료.\n")
        except Exception as e:
            # 학습 중 오류 발생 시, 오류 시점 모델 저장
            print(f"학습 중 오류 발생: {e}")
            model.save(os.path.join(model_path, "last_model_on_error.zip"))
            print(f"오류 발생 시점의 모델이 {os.path.join(model_path, 'last_model_on_error.zip')}에 저장되었습니다.")
            raise

        # 8. 최종 모델 저장
        model.save(model_path)
        print(f"최종 모델이 {model_path} 에 저장되었습니다.\n")
