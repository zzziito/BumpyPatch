import subprocess

original_pcd = '/home/rtlink/jiwon/cropped3.pcd'
output_name = "cropped3"


def run_script(script_path, args=[]):
    subprocess.run(["python3", script_path] + args)

if __name__ == "__main__":
    run_script("pc_to_heightmap_patch.py", [original_pcd, f'../result/{output_name}', "3.0", "15"])
    run_script("model_application.py", [f'../result/{output_name}', f'../result/{output_name}_label.csv'])
    run_script("make_label.py", [f'../result/{output_name}.csv', f'../result/{output_name}_label.csv',f'../result/{output_name}_final.csv'])
    run_script("pc_colorize.py", [original_pcd, f'../result/{output_name}_final.csv', f'../result/{output_name}_final.pcd'])
