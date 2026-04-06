import json

def calculate_speedup(file1_path, file2_path):
    with open(file1_path, 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)
    with open(file2_path, 'r', encoding='utf-8') as file2:
        data2 = json.load(file2)

    speedup_dict = {}
    total_speedup = 0
    valid_count = 0

    for key in data1:
        if key in data2:
            value1 = data1[key]
            value2 = data2[key]
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)) and value2 != 0:
                speedup = value1 / value2
                speedup_dict[key] = speedup
                total_speedup += speedup
                valid_count += 1

    average_speedup = total_speedup / valid_count if valid_count > 0 else 0
    return speedup_dict, average_speedup

file1_path = 'specee_awq.json'
file2_path = 'raw_awq.json'
speedup_dict, average_speedup = calculate_speedup(file1_path, file2_path)
print('Speedup ratio:')
for key, speedup in speedup_dict.items():
    print(f"{key}: {speedup:.4f}")

print(f"AWQ+SpecEE Average speedup: {average_speedup:.4f}")