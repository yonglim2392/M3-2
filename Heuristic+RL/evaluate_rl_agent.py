#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PROGRAM NAME  : evaluate_agent.py
PROGRAMMER    : YONG
CREATION DATE : 2025-08-29

[Description]
  - Stable-Baselines3 PPO 기반 학습된 RL 에이전트를 평가하는 스크립트
  - SchedulingEnv 환경에서 학습된 에이전트를 사용하여 주문 스케줄링 시뮬레이션 수행
  - 평가 중 액션 마스크 처리, 납기 준수, 보상 누적 등 에피소드 단위로 실행
  - 최종 스케줄을 납기 미준수 최소화 및 보상 최대 기준으로 선택 후 CSV 저장
"""

import os
import pandas as pd
from stable_baselines3 import PPO
from scheduling_env import SchedulingEnv
import numpy as np

def evaluate_agent(model_path, result_path, order_path, info_path, start_date_str, daily_work_hours, optimized_orders_df=None):
    """
    학습된 RL 에이전트 평가 함수

    Args:
        model_path (str): 저장된 RL 모델(.zip) 경로
        result_path (str): 평가 후 최적 스케줄을 저장할 CSV 경로
        order_path (str): 주문 CSV 파일 경로 (SchedulingEnv에 전달)
        info_path (str): 라인 정보 CSV 파일 경로 (SchedulingEnv에 전달)
        start_date_str (str): 환경 시작일 ("YYYY-MM-DD")
        daily_work_hours (int/float): 하루 작업시간 (예: 8)
        optimized_orders_df (pd.DataFrame, optional): (선택) 사전 최적화된 주문표 전달용

    동작:
        - 모델을 로드하고 SchedulingEnv 환경에서 한 에피소드(전체 주문 처리)를 deterministic하게 실행
        - 액션 마스크를 확인하여 invalid action 선택 시 보정
        - 에피소드 종료 후 스케줄 결과를 수집, 납기 미준수 수와 보상 기반으로 최종 스케줄을 선택하고 CSV로 저장
    """
    
    print("--- 학습된 RL 에이전트 평가 시작 ---")

    # 1) 모델 파일 존재 확인
    if not os.path.exists(model_path):
        print(f"오류: 학습된 모델을 찾을 수 없습니다: {model_path}")
        print("먼저 train_rl_agent.py를 실행하여 모델을 학습시켜야 합니다.")
        return

    # 2) 평가에 사용할 환경 생성
    #    SchedulingEnv 생성자에 optimized_orders_df 를 전달할 수 있도록 함수 시그니처에 포함되어 있음
    env = SchedulingEnv(
        order_path=order_path,
        info_path=info_path,
        start_date_str=start_date_str,
        daily_work_hours=daily_work_hours,
        optimized_orders_df=optimized_orders_df
    )

    # 3) 학습된 모델 로드 (CPU로 로드)
    try:
        model = PPO.load(model_path, device="cpu")
        print(f"모델 로드 성공: {model_path}")
    except Exception as e:
        # 로드 실패 시 에러 출력 후 종료
        print(f"모델 로드 중 오류 발생: {e}")
        return

    # --- 평가 상태 변수 초기화 ---
    best_schedule_df = None         # 평가 중 발견한 가장 좋은 스케줄 (DataFrame)
    best_late_orders = float('inf') # 최소 납기 미준수 개수 (작을수록 좋음)
    best_reward = -float('inf')     # 동률일 때 비교하는 보상 기준 (클수록 좋음)

    # 4) 환경 초기화: reset()으로 초기 관측(obs)과 info 획득
    obs, info = env.reset()
    done = False
    total_reward = 0

    # 5) 에피소드 루프: 환경이 종료(done)될 때까지 반복
    while not done:
        # (a) 정책(model)으로부터 행동 예측
        # deterministic=True -> 평가 시 동일 동작 재현을 위해 결정적 행동 사용
        action, _states = model.predict(obs, deterministic=True)

        # (b) 안전 점검: 관측값이 None이면 더 이상 진행 불가
        if obs is None:
            print("경고: 관측값이 None입니다. 에피소드를 종료합니다.")
            done = True
            continue

        # (c) 액션 마스크 처리
        #    - 환경의 상태 벡터 끝에 액션마스크를 붙여서 전달하는 구조라 가정
        #    - action_mask는 obs 배열의 마지막 num_lines 원소로 존재
        num_lines = env.get_num_actions()
        action_mask = obs[-num_lines:]

        # (d) 모델이 선택한 action이 마스크상 허용되지 않으면 보정
        #    - 정책은 환경에서 허용되지 않은 action을 낼 수 있음 (특히 학습 초반)
        #    - 허용된 action들(valid_actions)의 첫 번째를 fallback으로 선택 (간단한 보정)
        if action_mask[action] == 0:
            valid_actions = np.where(action_mask == 1)[0]
            if len(valid_actions) > 0:
                # 가능한 액션 중 첫 번째를 선택 (더 정교한 전략으로 변경 가능)
                action = valid_actions[0]
            else:
                # 허용된 행동이 전혀 없으면 에피소드 종료 (환경 설계에 따라 다르게 처리 가능)
                print(f"경고: 액션 마스크에 유효한 행동이 없습니다. 에피소드를 종료합니다.")
                done = True # Stop this episode if no valid action is possible
                continue

        # (e) 액션이 action_space 범위를 벗어나는 경우 방지
        if not (0 <= action < env.action_space.n):
            print(f"경고: 최종 선택된 행동({action})이 유효 범위를 벗어났습니다. 에피소드를 종료합니다.")
            done = True
            continue

        # 6) 선택한 액션으로 환경 한 스텝 진행
        #    - env.step() 반환값: next_obs, reward, terminated, truncated, info
        next_obs, reward, terminated, truncated, info = env.step(action)

        # 7) 종료 플래그 처리
        done = terminated or truncated

        # 8) 보상 누적
        total_reward += reward

        # 9) 관측값 갱신
        obs = next_obs

    # --- 에피소드 종료: 결과 분석 및 저장 처리 ---

    # 10) 환경이 기록한 스케줄 결과(DataFrame) 가져오기
    current_schedule_df = env.get_scheduled_df()

    # 11) 해당 스케줄에서 납기 미준수 건 수 집계(Missed_Delivery 컬럼이 존재하면 합산)
    current_late_orders = 0
    if not current_schedule_df.empty and 'Missed_Delivery' in current_schedule_df.columns:
        current_late_orders = current_schedule_df['Missed_Delivery'].sum()

    # 12) 현재 결과가 '최적'인지 판단하는 로직
    #    - 우선 기준: 납기 미준수 건수 (작을수록 좋음)
    #    - 동률일 경우: 누적 보상(total_reward) 기준 (클수록 좋음)
    is_new_best = False
    if current_late_orders < best_late_orders:
        is_new_best = True
    elif current_late_orders == best_late_orders and total_reward > best_reward:
        is_new_best = True

    # 13) 최적이라면 기록 업데이트
    if is_new_best:
        print(f"*** 새로운 최적 스케줄 발견! (납기 미준수: {current_late_orders}, 보상: {total_reward:.2f}) ***")
        best_late_orders = current_late_orders
        best_reward = total_reward
        best_schedule_df = current_schedule_df.copy()

    # 14) 최종 결과 출력 및 CSV 저장
    if best_schedule_df is not None:
        print("\n--- 최종 선택된 최적 스케줄 (일부) ---")
        # DataFrame을 마크다운 형태로 일부 출력 (가독성 위해)
        print(best_schedule_df.head().to_markdown(index=False))
        print(f"\n최종 스케줄된 작업 수: {len(best_schedule_df)}")
        print(f"최종 납기 미준수 건수: {best_late_orders}")
        print(f"최종 보상 점수: {best_reward:.2f}")

         # 결과 폴더가 없으면 생성 후 CSV 저장
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        best_schedule_df.to_csv(result_path, index=False)
        print(f"\n최적 스케줄링 결과가 {result_path} 에 저장되었습니다.")
    else:
        # best_schedule_df가 None이면 유효한 스케줄을 생성하지 못했다는 의미
        print("\n오류: 모든 평가 실행에서 유효한 스케줄을 생성하지 못했습니다.")
