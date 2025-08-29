#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PROGRAM NAME  : main.py
PROGRAMMER    : YONG
CREATION DATE : 2025-08-29

[Description]
  - RL 기반 생산 스케줄링 에이전트를 학습, 평가, (선택적으로) 시각화하는 메인 실행 스크립트
  - 사용자가 입력한 주문 데이터와 라인 정보를 기반으로 RL 모델(PPO)을 학습
  - 학습 완료 후 모델을 평가하고, 결과를 CSV로 저장
  - 필요 시 Gantt 차트 형태로 작업 일정 시각화 가능

사용 예시:
    python main.py --order_path ./DATA/TEST/YONGJIN2_ORDER.csv \
                   --info_path ./DATA/TEST/YONGJIN2_INFO.csv \
                   --start_date 2025-07-28 --work_hours 8

출력:
    - 학습된 PPO 모델 파일 (*.zip)
    - 평가 결과 CSV
    - (선택) 시각화 그래프
"""

import pandas as pd
import argparse
from train_rl_agent import train_agent
from evaluate_rl_agent import evaluate_agent
from make_plot import visualize_schedule

# python main.py --order_path ./DATA/TEST/YONGJIN2_ORDER.csv --info_path ./DATA/TEST/YONGJIN2_INFO.csv --start_date 2025-07-28 --work_hours 8

#*===========================================================
#* USER SET
#*===========================================================
# --- 사용자 설정: 로그 디렉토리와 학습 timestep ---
LOG_DIR          = './logs/' # 학습 로그 및 TensorBoard 저장 경로
TRAIN_TIMESTEPS  = 10000     # RL 학습 총 timestep
#*===========================================================

def main():
    # --- 1. 명령행 인자 처리 ---
    parser = argparse.ArgumentParser(description="RL-based Production Scheduling Agent")
    
    parser.add_argument('--order_path', type=str, default='./DATA/YONGJIN2_ORDER.csv', help='Path to the order data CSV file.')
    parser.add_argument('--info_path',  type=str, default='./DATA/YONGJIN2_INFO.csv',  help='Path to the line info data CSV file.')
    parser.add_argument('--start_date', type=str, default='2025-01-01',                help='Scheduling start date (YYYY-MM-DD).')
    parser.add_argument('--work_hours', type=int, default=10,                          help='Daily work hours.')

    args = parser.parse_args()

    # --- 2. 라인 정보 로딩 및 모델/결과 경로 설정 ---
    INFO = pd.read_csv(args.info_path)
    LINE_NUMBER      = len(INFO) # 라인 수
    MODEL_PATH       = f'./RL_MODEL/ppo_scheduling_model_L{LINE_NUMBER}.zip'                 # 학습된 모델 저장 경로
    RESULT_SAVE_PATH = f'./RESULT/rl_scheduled_result_L{LINE_NUMBER}({args.start_date}).csv' # 평가 결과 저장 경로

    # --- 3. RL 에이전트 학습 ---
    train_agent(
        model_path       = MODEL_PATH,
        logs_dir         = LOG_DIR,
        order_path       = args.order_path,
        info_path        = args.info_path,
        start_date_str   = args.start_date,
        daily_work_hours = args.work_hours,
        train_timesteps  = TRAIN_TIMESTEPS
        )

    # --- 4. 학습된 모델 평가 및 결과 저장 ---
    evaluate_agent(
        model_path       = MODEL_PATH,
        result_path      = RESULT_SAVE_PATH,
        order_path       = args.order_path,
        info_path        = args.info_path,
        start_date_str   = args.start_date,
        daily_work_hours = args.work_hours
        )

    # --- 5. (선택) 작업 일정 시각화 --
    #visualize_schedule(data_path = RESULT_SAVE_PATH)
    
# --- 스크립트 직접 실행 시 main() 호출 ---
if __name__ == "__main__":
    main()
