
import pandas as pd
import argparse
from train_rl_agent import train_agent
from evaluate_rl_agent import evaluate_agent
from make_plot import visualize_schedule

# python main.py --order_path ./DATA/TEST/YONGJIN2_ORDER.csv --info_path ./DATA/TEST/YONGJIN2_INFO.csv --start_date 2025-07-28 --work_hours 8

#*===========================================================
#* USER SET
#*===========================================================
LOG_DIR          = './logs/'
TRAIN_TIMESTEPS  = 10000
#*===========================================================

def main():
    parser = argparse.ArgumentParser(description="RL-based Production Scheduling Agent")
    
    parser.add_argument('--order_path', type=str, default='./DATA/YONGJIN2_ORDER.csv', help='Path to the order data CSV file.')
    parser.add_argument('--info_path',  type=str, default='./DATA/YONGJIN2_INFO.csv',  help='Path to the line info data CSV file.')
    parser.add_argument('--start_date', type=str, default='2025-01-01',                help='Scheduling start date (YYYY-MM-DD).')
    parser.add_argument('--work_hours', type=int, default=10,                          help='Daily work hours.')

    args = parser.parse_args()

    INFO = pd.read_csv(args.info_path)
    LINE_NUMBER      = len(INFO)
    MODEL_PATH       = f'./RL_MODEL/ppo_scheduling_model_L{LINE_NUMBER}.zip'
    RESULT_SAVE_PATH = f'./RESULT/rl_scheduled_result_L{LINE_NUMBER}({args.start_date}).csv'
    
    train_agent(
        model_path       = MODEL_PATH,
        logs_dir         = LOG_DIR,
        order_path       = args.order_path,
        info_path        = args.info_path,
        start_date_str   = args.start_date,
        daily_work_hours = args.work_hours,
        train_timesteps  = TRAIN_TIMESTEPS
        )

    evaluate_agent(
        model_path       = MODEL_PATH,
        result_path      = RESULT_SAVE_PATH,
        order_path       = args.order_path,
        info_path        = args.info_path,
        start_date_str   = args.start_date,
        daily_work_hours = args.work_hours
        )
    
    #visualize_schedule(data_path = RESULT_SAVE_PATH)
    

if __name__ == "__main__":
    main()
