import pandas as pd
import  os


def process_sensorlog_csv(path):
    print("📋 文件列名：", pd.read_csv(path, nrows=1).columns.tolist())
    df = pd.read_csv(path, usecols=[
        'accelerometerTimestamp_sinceReboot(s)',
        'accelerometerAccelerationX(G)',
        'accelerometerAccelerationY(G)',
        'accelerometerAccelerationZ(G)',
    ])
    df = df.rename(columns={
        'accelerometerTimestamp_sinceReboot(s)': 'seconds',
        'accelerometerAccelerationX(G)': 'x',
        'accelerometerAccelerationY(G)': 'y',
        'accelerometerAccelerationZ(G)': 'z',
    })
    return df[['seconds', 'x', 'y', 'z']]

def trim_data(df, start, end):
   df = df[(df['seconds'] >= start) & (df['seconds'] <= end)]
   df['seconds'] = df['seconds'] - start
   return df

def main():
    os.makedirs('processed_data', exist_ok=True)

    files = {
        'y_hand1': 'Y-hand(1).csv',
        # 'y_pocket1': 'Y-pocket(1).csv',
        # 'y_pocket2': 'Y-pocket(2).csv',
        'y_pocket3': 'Y-pocket(3).csv',
        'y_pocket4': 'Y-pocket(4).csv',
        # 'hand': 'hand.csv',
        # 'pocket': 'pocket.csv',
        # 'ankle': 'ankle.csv',
    }

    for outname , filename in files.items():
        df = process_sensorlog_csv(f'../data/{filename}')
        if df['seconds'].iloc[-1]  >= 30:
            df.to_csv(f'processed_data/{filename}', index=False)
        else:
            print(f'skipping {filename}: too short')

if __name__ == '__main__':
    main()