import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

def parse_srt_block(content: str) -> List[str]:
    """
    解析 SRT 内容，提取纯文本。
    策略：
    1. 按照空行分割字幕块。
    2. 忽略纯数字行（序号）。
    3. 忽略包含 '-->' 的行（时间轴）。
    4. 移除 HTML 标签 (<i>, <b>, <font>)。
    5. 将同一个字幕块内的多行文本合并为一行（方便 LQA 对齐）。
    """
    # 统一换行符
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    # 按双换行符分割字幕块
    blocks = content.split('\n\n')
    
    cleaned_lines = []
    
    # 正则：匹配 HTML 标签
    re_html = re.compile(r'<[^>]+>')
    # 正则：匹配 ASS/SSA 花括号代码 (偶尔出现在 SRT 中)
    re_ass = re.compile(r'\{[^}]+\}')

    for block in blocks:
        lines = block.strip().split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            # 跳过纯数字序号
            if line.isdigit():
                continue
            # 跳过时间轴
            if '-->' in line:
                continue
            
            # 清洗标签
            line = re_html.sub('', line)
            line = re_ass.sub('', line)
            
            if line:
                text_lines.append(line)
        
        if text_lines:
            # 将该块内的多行文本用空格连接，变成单行
            # 例如：
            # "Hello"
            # "World"
            # 变为 "Hello World"
            combined_text = ' '.join(text_lines)
            cleaned_lines.append(combined_text)
            
    return cleaned_lines

def convert_file(input_path: str, output_path: str = None) -> int:
    path = Path(input_path)
    if not path.exists():
        print(f"❌ 错误: 文件未找到 - {input_path}")
        return 0

    try:
        # 尝试检测编码，通常是 utf-8 或 utf-8-sig
        with open(path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            # 尝试 gbk (针对老旧中文字幕)
            with open(path, 'r', encoding='gbk') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ 编码错误，无法读取文件: {e}")
            return 0

    lines = parse_srt_block(content)
    
    # 如果没指定输出路径，默认生成 .txt
    if not output_path:
        output_path = path.with_suffix('.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        
    print(f"✅ 已转换: {path.name} -> {Path(output_path).name} ({len(lines)} 行)")
    return len(lines)

def main():
    parser = argparse.ArgumentParser(description="SRT 字幕清洗工具 - 配合 LQA 使用")
    parser.add_argument("files", nargs='+', help="需要转换的 .srt 文件路径 (支持多个)")
    parser.add_argument("--pair", action="store_true", help="启用成对检查模式 (检查双语行数是否一致)")
    
    args = parser.parse_args()
    
    counts = []
    for file_path in args.files:
        count = convert_file(file_path)
        counts.append(count)

    # 如果开启了 Pair 模式，且输入了两个文件，检查行数是否对其
    if args.pair and len(counts) == 2:
        print("-" * 30)
        if counts[0] == counts[1]:
            print("🟢 完美！两个文件的行数完全一致，可以直接放入 LQA 工具。")
        else:
            print(f"🔴 警告：行数不匹配！({counts[0]} vs {counts[1]})")
            print("这可能导致 LQA 工具错位。请检查原字幕文件的时间轴合并情况。")

if __name__ == "__main__":
    main()
