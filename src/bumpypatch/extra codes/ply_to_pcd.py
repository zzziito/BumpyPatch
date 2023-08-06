import open3d as o3d
import os

# PLY 파일이 있는 디렉토리 경로
input_directory = '/media/rtlink/JetsonSSD-256/Download/Rellis_3D_os1_cloud_node_color_ply/Rellis-3D/00000/os1_cloud_node_color_ply'

# 출력할 PCD 파일의 디렉토리 (입력 디렉토리와 동일하게 설정할 수 있음)
output_directory = input_directory

# 디렉토리 내의 모든 파일 조회
for filename in os.listdir(input_directory):
    # PLY 파일만 처리
    if filename.endswith('.ply'):
        # 입력 파일 경로
        input_path = os.path.join(input_directory, filename)
        
        # 출력 파일 경로 (확장자를 .pcd로 변경)
        output_path = os.path.join(output_directory, os.path.splitext(filename)[0] + '.pcd')
        
        # PLY 파일 읽기
        cloud = o3d.io.read_point_cloud(input_path)
        
        # PCD 파일로 저장
        o3d.io.write_point_cloud(output_path, cloud)
        print(f'Converted {filename} to {os.path.basename(output_path)}')
        
print("Conversion complete!")
