import pandas as pd
import argparse

def assign_class(row):
    if row['label'] == 'high_noise':
        if 0 <= row['z_diff'] < 1.5:
            return 2
        elif 1.5 <= row['z_diff'] < 2:
            return 3
        elif row['z_diff'] >= 2:
            return 4
    elif row['label'] == 'low_noise':
        if 0 <= row['z_diff'] < 0.5:
            return 1
        elif 0.5 <= row['z_diff'] < 2.5:
            return 2
        elif row['z_diff'] >= 2.5:
            return 4
    elif row['label'] == 'noise_sin':
        if 0 <= row['z_diff'] < 0.5:
            return 1
        elif 0.5 <= row['z_diff'] < 1:
            return 2
        elif 1 <= row['z_diff'] < 2.5:
            return 3
        elif row['z_diff'] >= 2.5:
            return 4
    elif row['label'] == 'flat':
        if 0 <= row['z_diff'] < 2:
            return 1
        elif row['z_diff'] >= 2:
            return 4
    elif row['label'] == 'none':
        return 4
    else:
        return None  # 알 수 없는 라벨은 None 반환

def process_csv(input_file1, input_file2, output_file):
    # CSV 파일 읽기
    patch_info = pd.read_csv(input_file1)
    patch_label = pd.read_csv(input_file2)

    # 'patch'와 'image'가 일치하는 행끼리 결합
    patch_info = pd.merge(patch_info, patch_label, left_on='patch', right_on='image')

    # 'image' 열은 이제 필요 없으므로 삭제
    patch_info = patch_info.drop('image', axis=1)

    # CSV 파일로 다시 저장
    patch_info.to_csv(output_file, index=False)

    # 'max_z'와 'min_z'의 차이 계산
    patch_info['z_diff'] = patch_info['max_z'] - patch_info['min_z']

    # CSV 파일로 다시 저장
    patch_info.to_csv(output_file, index=False)

    # 새로운 'class' 열 생성
    patch_info['class'] = patch_info.apply(assign_class, axis=1)

    # CSV 파일로 다시 저장
    patch_info.to_csv(output_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process csv files')
    parser.add_argument('input_file1', help='First input csv file')
    parser.add_argument('input_file2', help='Second input csv file')
    parser.add_argument('output_file', help='Output csv file')
    args = parser.parse_args()

    process_csv(args.input_file1, args.input_file2, args.output_file)
