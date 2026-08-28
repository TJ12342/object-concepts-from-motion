import argparse
from tbparse import SummaryReader
import pandas as pd
import os

def get_scalar_value_at_last_step(log_dir, scalar_tag):
    """
    从 TensorBoard 日志中提取指定标量指标在最后一个记录步骤的值。

    参数:
    log_dir (str): TensorBoard 日志文件所在的目录。
    scalar_tag (str): TensorBoard 中记录标量指标时使用的标签名称。

    返回:
    tuple (value, step) or (None, None): 包含最后一个步骤的值和对应步数的元组，
                                         如果找不到指标或数据则返回 (None, None)。
    """
    try:
        # 检查日志目录是否存在
        if not os.path.isdir(log_dir):
            print(f"错误: 日志目录 '{log_dir}' 不存在。")
            return None, None

        reader = SummaryReader(log_dir, extra_columns={'dir_name'})
        df = reader.scalars

        if df.empty:
            return None, None

        # 筛选出指定的标量指标
        scalar_df = df[df['tag'] == scalar_tag].copy() # 使用 .copy() 避免 SettingWithCopyWarning

        if scalar_df.empty:
            return None, None

        if scalar_df['value'].isnull().all():
            print(f"警告: 标签 '{scalar_tag}' 的所有值都为空 (NaN)。")
            return None, None

        # 确保 'step' 列是数值类型，以正确排序
        scalar_df['step'] = pd.to_numeric(scalar_df['step'])

        # 按 'step' 排序并获取最后一个（最大 step）
        last_step_row = scalar_df.sort_values(by='step', ascending=True).iloc[-1]

        return last_step_row['value'], last_step_row['step']

    except FileNotFoundError:
        print(f"错误: 日志目录 '{log_dir}' 中的事件文件未找到 (可能是空的或路径问题)。")
        return None, None
    except IndexError: # 如果 scalar_df 筛选后为空，iloc[-1] 会引发 IndexError
        # print(f"信息: 在日志中没有找到标签为 '{scalar_tag}' 的有效数据点后进行排序。")
        return None, None
    except Exception as e:
        print(f"处理日志目录 '{log_dir}' 中标签 '{scalar_tag}' 时发生错误: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="从 TensorBoard 日志中提取指定指标在最后一个记录步骤的值。")
    parser.add_argument("log_dir", type=str, help="TensorBoard 日志文件所在的目录。")
    parser.add_argument(
        "--tags",
        type=str,
        nargs='+',
        default=["val/RayIoU", "val/RayIoU@1", "val/RayIoU@2", "val/RayIoU@4"],
        help="要提取的指标标签列表。默认为: val/RayIoU val/RayIoU@1 val/RayIoU@2 val/RayIoU@4"
    )

    args = parser.parse_args()

    print(f"正在从日志目录 '{args.log_dir}' 中提取数据...")
    print(f"目标指标 (最后一个步骤的值): {', '.join(args.tags)}")
    print("-" * 30)

    results = {}
    all_tags_found_in_logs = set()

    try:
        # 尝试读取一次以获取所有可用的tag，用于更友好的提示
        # verbose=False 减少 tbparse 不必要的标准输出
        reader_check = SummaryReader(args.log_dir, verbose=False)
        if not reader_check.scalars.empty:
            all_tags_found_in_logs = set(reader_check.scalars['tag'].unique())
    except Exception as e:
        # print(f"初步检查日志时发生警告: {e}") # 对于批量处理，可能不需要此信息
        # 如果初始读取失败（例如目录无效），get_scalar_value_at_last_step 会处理
        pass


    for tag_name in args.tags:
        last_value, at_step = get_scalar_value_at_last_step(args.log_dir, tag_name)
        if last_value is not None and at_step is not None:
            results[tag_name] = {"value": last_value, "step": at_step}
            print(f"指标: {tag_name}, 最后步骤 ({int(at_step)}) 的值: {last_value:.3f}")
        else:
            if not all_tags_found_in_logs or tag_name not in all_tags_found_in_logs:
                 print(f"指标: {tag_name}, 未在日志中找到该标签。")
            else:
                 print(f"指标: {tag_name}, 找到了标签但无法提取最后一个步骤的数据点 (可能为空或全为NaN)。")


    if not results:
        print("-" * 30)
        print("未能从指定的日志目录中提取任何请求的指标数据。")
        if all_tags_found_in_logs:
            print("\n在日志中找到的可用标量标签包括:")
            for i, found_tag in enumerate(sorted(list(all_tags_found_in_logs))):
                print(f" - {found_tag}")
                if i > 20 and len(all_tags_found_in_logs) > 25: # 避免打印过多
                    print(f" ... 以及其他 {len(all_tags_found_in_logs) - i -1} 个标签。")
                    break
        else:
            print("日志目录中似乎没有任何标量数据，或者目录路径不正确/无法访问。")
    else:
        for tag, values in results.items():
            value = values["value"] * 100
            print(f"& {value:.1f}", end=" ")
        print("\\\\\n")

    # 你也可以选择将结果保存到文件等
    # 例如，保存为 JSON
    # import json
    # if results:
    #     with open("last_step_rayiou_results.json", "w") as f:
    #         json.dump(results, f, indent=4)
    #     print("\n结果也已保存到 last_step_rayiou_results.json")

if __name__ == "__main__":
    # 确保在运行前已安装 tbparse 和 pandas
    # pip install tbparse pandas
    main()