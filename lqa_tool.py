import sys
import json
import re
import pysubs2
from typing import List, Dict

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

import pysubs2

def parse_subtitle_file(file_path):
    """
    通用解析函数：支持 SRT, ASS, VTT。
    """
    try:
        subs = pysubs2.load(file_path)
        parsed = []
        for line in subs:
            # 1. 基础清理
            text = line.plaintext.strip()
            
            # 2. ASS 特殊处理：将 ASS 的硬换行 \N 替换为看起来的换行，避免连在一起
            text = text.replace(r"\N", "\n").replace(r"\n", "\n")
            
            parsed.append({
                'start': line.start, # 毫秒
                'end': line.end,     # 毫秒
                'text': text
            })
        return parsed
    except Exception as e:
        print(f"解析出错 {file_path}: {e}")
        return []

def align_subtitles(source_data, target_data):
    """
    基于时间轴的严格对齐算法。
    避免将不相关的行强行合并。
    """
    aligned = []
    
    # 最小有效重叠时间 (毫秒)，小于这个时间的重叠忽略不计
    MIN_OVERLAP_MS = 200 
    
    # 用于标记哪些译文已经被使用过，防止重复分配
    used_target_indices = set()

    for s in source_data:
        s_start = s['start']
        s_end = s['end']
        
        # 寻找属于当前原文的所有译文候选
        candidates = []
        
        for t_idx, t in enumerate(target_data):
            if t_idx in used_target_indices:
                continue
                
            t_start = t['start']
            t_end = t['end']
            
            # 计算重叠部分
            overlap_start = max(s_start, t_start)
            overlap_end = min(s_end, t_end)
            overlap_duration = overlap_end - overlap_start
            
            # 译文自身的持续时间
            t_duration = t_end - t_start
            if t_duration <= 0: t_duration = 1 # 防止除以0
            
            # 判定标准：
            # 1. 重叠时间必须 > 200ms (防止仅仅边缘擦过)
            # 2. 或者：重叠部分占了译文总时长的 50% 以上 (说明这句话大部分时间都在这行原文里)
            is_valid_match = (overlap_duration >= MIN_OVERLAP_MS) or \
                             (overlap_duration / t_duration > 0.5)
            
            if is_valid_match:
                candidates.append((t_idx, t['text']))

        # 整理结果
        if candidates:
            # 按索引排序，保证文本顺序
            candidates.sort(key=lambda x: x[0])
            
            # 【关键修改】使用换行符连接，而不是空格，这样在表格里能看出来是多行
            combined_text = "\n".join([c[1] for c in candidates])
            
            aligned.append((s['text'], combined_text))
            
            # 标记这些译文已被消耗
            for c in candidates:
                used_target_indices.add(c[0])
        else:
            # 没有匹配到译文，留空
            aligned.append((s['text'], ""))
            
    # 【可选】检查是否有剩下的译文（原文没覆盖到的），追加到最后（防止漏译文）
    # 如果你想把剩下的译文也显示出来，取消下面注释
    # for t_idx, t in enumerate(target_data):
    #     if t_idx not in used_target_indices:
    #         aligned.append(("[无原文匹配]", t['text']))

    return aligned


from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QMenu, QTableWidget, QTableWidgetItem, 
                               QPushButton, QHeaderView, QLabel, QSplitter, 
                               QFileDialog, QProgressBar, QMessageBox, QLineEdit,
                               QPlainTextEdit, QFrame, QComboBox, QCheckBox)
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QColor, QBrush, QFont

from google import genai
from google.genai import types

# --- 配置与 Prompts ---

# 对齐 Prompt：设计为通用型，无论以谁为主都能工作
ALIGN_PROMPT = """
### Role
You are a strictly mechanical Subtitle Resegmentation Engine.
Your ONLY job is to take text from the "Source Pool" and cut/combine it to match the number of lines and semantic flow of the "Master Lines".

### Rules (CRITICAL)
1. **NO TRANSLATION**: You must output the text exactly as it appears in the Source Pool (same language, same words).
2. **STRICT COUNT**: The output JSON array MUST have exactly the same number of elements as the "Master Lines".
3. **NO HALLUCINATION**: Do not invent content. If the Source Pool is empty or lacks corresponding text for a line, return an empty string "".
4. **SEQUENCE**: The text in the Source Pool generally follows the time order of the Master Lines.

### Task
Input:
- Master Lines (The reference timeline/segments).
- Source Pool (The chaotic text text that needs re-segmenting).

Output:
- A JSON Array of strings.

### Example
**Input**:
Master (Target): ["Good morning.", "How are you?"]
Pool (Source): "早安你好吗"
**Output**:
["早安", "你好吗"]

**Input**:
Master: ["Wait.", "I...", "I didn't mean it."]
Pool: "等等我我不是那个意思"
**Output**:
["等等", "我...", "我不是那个意思"]
"""

LQA_PROMPT = """
# Role
你是一名拥有20年经验的资深本地化质量保证专家（LQA Specialist）。你以极度严苛、绝不妥协，对文字极其敏感，无法容忍任何形式的平庸翻译。你的目标是确保所有译文不仅准确，而且在目标语言中达到出版级母语水准。

# Task
我将提供给你一组“原文（Source）”和“译文（Target）”。请你逐句对译文进行深度的LQA审查。

# Evaluation Criteria
你需要从以下四个维度进行无情地批判：
1. **错译**：逻辑错误、术语错误、对原文理解偏差。
2. **漏译**：丢失了原文的关键信息、修饰语或语气。
3. **过度意译**：脱离原文太远，添加了原文没有的含义，或者风格不符。
4. **生硬/翻译腔**：句式欧化（或源语言化）、选词不地道、读起来像机器翻译或蹩脚的直译。

# Tone and Style
- **严厉直接**：不要使用“整体不错”、“还可以”这种客套话。如果翻译得很烂，直接指出。
- **一针见血**：精准定位问题所在，不要模糊其词。
- **专业**：使用语言学或翻译理论术语（如：句法干扰、语义丢失等）来辅助你的批评。

# Output Format
# Output Format (MANDATORY JSON)
你必须输出一个纯 JSON 数组。不要使用 Markdown 格式，不要输出 ```json 代码块。
数组中的每个对象必须包含以下字段：
- "id": (整数) 对应输入的 ID。
- "score": (整数) 0-10 分。
- "issues": (字符串列表) 如 ["错译", "生硬"]，无问题则为空列表 []。
- "comment": (字符串) 你的严厉点评内容。
- "suggestion": (字符串) 你修改后的高质量译文。
---
"""

# --- 工具函数 ---
def clean_ass_text(text: str) -> str:
    r"""
    强力清洗 ASS/SSA 标签
    1. 移除 {} 内的所有内容 (样式代码)
    2. 移除 \N, \n, \h 等转义符
    3. 移除绘图指令等杂项
    """
    # 1. 移除 ASS 标签 { ... }
    text = re.sub(r'\{[^}]+\}', '', text)
    # 2. 移除常见的转义换行
    text = text.replace(r'\N', ' ').replace(r'\n', ' ').replace(r'\h', ' ')
    # 3. 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 智能对齐器 (支持双向) ---
class AutoAligner(QThread):
    progress_update = Signal(int, int, str)
    finished = Signal(list) 
    error_occurred = Signal(str) # 新增：错误信号

    def __init__(self, api_key, source_subs, target_subs, mode='source_master'):
        super().__init__()
        self.api_key = api_key
        self.source_subs = source_subs
        self.target_subs = target_subs
        self.mode = mode 
        self.batch_size = 8

    def run(self):
        logging.info("LQA Worker started.")
        total = len(self.source_lines)
        
        try:
            client = genai.Client(api_key=self.api_key)
            
            for i in range(0, total, self.batch_size):
                # 修正：使用驼峰命名法 isInterruptionRequested()
                if self.isInterruptionRequested(): 
                    logging.info("LQA Worker interrupted by user.")
                    break
                
                batch_s = self.source_lines[i : i + self.batch_size]
                batch_t = self.target_lines[i : i + self.batch_size]
                
                prompt_content = {
                    "source": batch_s,
                    "target": batch_t,
                    "start_index": i
                }
                
                prompt_str = json.dumps(prompt_content, ensure_ascii=False)
                
                self.progress_update.emit(i, total, f"Checking {i}/{total}...")
                
                try:
                    logging.debug(f"Sending LQA Batch {i}...")
                    response = client.models.generate_content(
                        model="gemini-pro-latest",
                        config=types.GenerateContentConfig(
                            system_instruction=LQA_PROMPT,
                            temperature=0.8,
                            response_mime_type="application/json"
                        ),
                        contents=[prompt_str]
                    )
                    
                    logging.debug(f"LQA Batch {i} Raw Response: {response.text}")
                    
                    try:
                        res_json = json.loads(response.text)
                        
                        if isinstance(res_json, dict) and "reviews" in res_json:
                             res_json = res_json["reviews"]
                        elif not isinstance(res_json, list):
                             res_json = []
                             
                        for idx, item in enumerate(res_json):
                            real_row_id = i + idx 
                            item['id'] = real_row_id
                            
                        self.batch_finished.emit(res_json)
                        
                    except json.JSONDecodeError:
                        logging.error(f"JSON Parse Error in Batch {i}")
                        
                except Exception as e:
                    logging.error(f"API Error in Batch {i}: {e}")
                    continue
            
            self.finished.emit()

        except Exception as e:
            logging.critical(f"LQA Worker Critical Error: {e}", exc_info=True)
            self.error_occurred.emit(str(e))

# --- LQA 执行器 ---
class LQAWorker(QThread):
    # PySide6 使用 Signal，而不是 pyqtSignal
    progress_update = Signal(int, int, str)
    batch_finished = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, api_key, source_lines, target_lines):
        super().__init__()
        self.api_key = api_key
        self.source_lines = source_lines
        self.target_lines = target_lines
        self.batch_size = 10 

    def run(self):
        logging.info("LQA Worker started.")
        total = len(self.source_lines)
        
        try:
            client = genai.Client(api_key=self.api_key)
            
            for i in range(0, total, self.batch_size):
                # PySide6/Qt 标准驼峰命名
                if self.isInterruptionRequested():
                    logging.info("LQA Worker interrupted by user.")
                    break
                
                batch_s = self.source_lines[i : i + self.batch_size]
                batch_t = self.target_lines[i : i + self.batch_size]
                
                prompt_content = {
                    "source": batch_s,
                    "target": batch_t,
                    "start_index": i
                }
                
                prompt_str = json.dumps(prompt_content, ensure_ascii=False)
                
                self.progress_update.emit(i, total, f"Checking {i}/{total}...")
                
                try:
                    logging.debug(f"Sending LQA Batch {i}...")
                    response = client.models.generate_content(
                        model="gemini-pro-latest",
                        config=types.GenerateContentConfig(
                            system_instruction=LQA_PROMPT,
                            temperature=1.0,
                            response_mime_type="application/json"
                        ),
                        contents=[prompt_str]
                    )
                    
                    try:
                        res_json = json.loads(response.text)
                        
                        if isinstance(res_json, dict) and "reviews" in res_json:
                             res_json = res_json["reviews"]
                        elif not isinstance(res_json, list):
                             res_json = []
                             
                        for idx, item in enumerate(res_json):
                            real_row_id = i + idx 
                            item['id'] = real_row_id
                            
                        self.batch_finished.emit(res_json)
                        
                    except json.JSONDecodeError:
                        logging.error(f"JSON Parse Error in Batch {i}")
                        
                except Exception as e:
                    logging.error(f"API Error in Batch {i}: {e}")
                    continue
            
        except Exception as e:
            logging.critical(f"LQA Worker Critical Error: {e}", exc_info=True)
            self.error_occurred.emit(str(e))



# --- 主界面 ---
class LQAModernWindowV3(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kaorou Checker (Time-Aware Alignment)")
        self.resize(1200, 800)
        
        # 保存文件路径，用于时间轴解析
        self.source_path = None
        self.target_path = None
        
        self.source_subs_obj = [] 
        self.target_subs_obj = []
        self.full_results = {}
        
        # 这里必须用 QSettings (如果你代码头没有导入，请确保导入 QSettings)
        from PySide6.QtCore import QSettings
        self.settings = QSettings("Kaorou", "LQATool")

        self.setup_ui()
        self.apply_styles()

        # 启用右键菜单
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)


    def setup_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)

        # 1. 顶部栏
        top_bar = QHBoxLayout()
        
        self.api_input = QLineEdit()
        self.api_input.setPlaceholderText("Gemini API Key")
        self.api_input.setEchoMode(QLineEdit.Password)
        self.api_input.setText(self.settings.value("api_key", ""))
        self.api_input.setFixedWidth(150)
        
        btn_src = QPushButton("📂 原文")
        btn_src.clicked.connect(lambda: self.load_file('source'))
        btn_tgt = QPushButton("📂 译文")
        btn_tgt.clicked.connect(lambda: self.load_file('target'))
        
        # 模式选择
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["以原文为轴 (Source Master)", "以译文为轴 (Target Master)"])
        self.combo_mode.setToolTip("选择谁是基准。如果你的译文是调过轴的最终版，请选'以译文为轴'")
        self.combo_mode.currentIndexChanged.connect(self.try_time_alignment)

        self.btn_action = QPushButton("🚀 分析")
        self.btn_action.clicked.connect(self.run_process)
        self.btn_action.setEnabled(False)

        top_bar.addWidget(QLabel("Key:"))
        top_bar.addWidget(self.api_input)
        top_bar.addWidget(btn_src)
        top_bar.addWidget(btn_tgt)
        top_bar.addWidget(QLabel("模式:"))
        top_bar.addWidget(self.combo_mode)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_action)
        layout.addLayout(top_bar)

        # 状态栏
        self.lbl_status = QLabel("请加载文件 (.srt, .ass, .vtt)")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)

        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)

        # 2. 分割视图
        splitter = QSplitter(Qt.Vertical)
        
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ID", "原文 (Source)", "译文 (Target)", "评分", "问题", "建议"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(3, 60)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemClicked.connect(self.on_row_clicked)
        splitter.addWidget(self.table)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        splitter.addWidget(self.detail_text)
        
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

    def apply_styles(self):
        self.setStyleSheet("""
            /* 全局白底黑字 */
            QMainWindow { background-color: #ffffff; color: #202124; }
            QWidget { font-family: "Segoe UI", sans-serif; font-size: 13px; color: #202124; }
            
            /* 输入框、表格：白底灰边 */
            QLineEdit, QComboBox, QPlainTextEdit, QTableWidget { 
                background-color: #ffffff; 
                color: #202124; 
                border: 1px solid #dadce0; 
                padding: 4px;
                selection-background-color: #d2e3fc; /* 选中时的淡蓝色 */
                selection-color: #202124;
            }
            
            /* 按钮：蓝色 */
            QPushButton { 
                background-color: #1a73e8; 
                color: #ffffff; 
                border-radius: 4px; 
                padding: 6px 12px; 
                font-weight: bold; 
                border: none;
            }
            QPushButton:hover { background-color: #1557b0; }
            QPushButton:disabled { background-color: #f1f3f4; color: #80868b; }
            
            /* 表头：浅灰背景 */
            QHeaderView::section { 
                background-color: #f1f3f4; 
                color: #202124; 
                border: 1px solid #dadce0; 
                font-weight: bold;
            }
            
            /* 表格选中行：Google 风格淡蓝 */
            QTableWidget::item:selected { 
                background-color: #e8f0fe; 
                color: #1967d2; 
            }
            
            /* 进度条 */
            QProgressBar { 
                text-align: center; 
                color: black; 
                border: 1px solid #dadce0; 
                background: #f1f3f4; 
                border-radius: 4px; 
            }
            QProgressBar::chunk { background-color: #1a73e8; }
        """)


    def load_file(self, file_type):
        """
        加载文件并触发对齐逻辑
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择文件", 
            "", 
            "Subtitle Files (*.srt *.ass *.vtt);;All Files (*)"
        )

        if not file_path:
            return

        if file_type == 'source':
            self.source_path = file_path
            logging.info(f"Loaded Source: {file_path}")
        else:
            self.target_path = file_path
            logging.info(f"Loaded Target: {file_path}")

        # 直接调用对齐，不要在这里尝试访问 aligned_data
        # 状态更新和 UI 刷新已经在 try_time_alignment 内部完成了
        self.try_time_alignment()



    def try_time_alignment(self):
        """ 
        核心：基于时间轴的对齐逻辑 
        根据 combo_mode 的选择决定是以原文还是译文为基准
        """
        # 1. 检查文件是否已加载
        if not self.source_path or not self.target_path:
            # 如果还没加载文件，只更新状态文字
            s_status = "已加载" if self.source_path else "未加载"
            t_status = "已加载" if self.target_path else "未加载"
            self.lbl_status.setText(f"就绪状态: 原文[{s_status}] | 译文[{t_status}]")
            return

        try:
            self.lbl_status.setText("正在进行时间轴智能对齐...")
            
            # 2. 解析文件 (利用缓存的路径)
            s_data = parse_subtitle_file(self.source_path)
            t_data = parse_subtitle_file(self.target_path)
            
            if not s_data or not t_data:
                QMessageBox.warning(self, "警告", "文件解析为空，请检查文件内容。")
                return

            # 3. 根据下拉框决定对齐方向
            # Index 0: 以原文为轴 (Source Master)
            # Index 1: 以译文为轴 (Target Master)
            is_target_master = (self.combo_mode.currentIndex() == 1)

            if is_target_master:
                # 【模式 B：以译文为轴】
                # 逻辑：遍历译文行，去原文里找对应的句子
                # align_subtitles 返回的是 List[(Axis_Text, Reference_Text)]
                # 所以这里得到的是 [(Target_Text, Source_Text_Combined), ...]
                raw_aligned = align_subtitles(t_data, s_data)
                
                # 转换数据以便显示：
                # 表格习惯是：左边原文(Col 1)，右边译文(Col 2)
                # 所以我们要把结果 tuple 翻转一下： (Source, Target)
                final_display_data = []
                for tgt_text, src_text in raw_aligned:
                    final_display_data.append((src_text, tgt_text))
                    
            else:
                # 【模式 A：以原文为轴 (默认)】
                # 逻辑：遍历原文行，去译文里找对应的句子
                # 得到 [(Source_Text, Target_Text_Combined), ...]
                raw_aligned = align_subtitles(s_data, t_data)
                
                # 顺序已经是 (Source, Target)，直接用
                final_display_data = raw_aligned

            # 4. 渲染到表格
            self.table.setRowCount(0)
            self.table.setRowCount(len(final_display_data))
            
            for row_idx, (s_text, t_text) in enumerate(final_display_data):
                # ID
                self.table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
                # 原文 (Column 1)
                self.table.setItem(row_idx, 1, QTableWidgetItem(s_text))
                # 译文 (Column 2)
                self.table.setItem(row_idx, 2, QTableWidgetItem(t_text))

            # 5. 调整 UI 状态
            self.table.resizeRowsToContents() 
            self.lbl_status.setText(f"对齐完成! 共 {len(final_display_data)} 行。")
            self.btn_action.setEnabled(True)
            self.table.viewport().update()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.lbl_status.setText(f"错误: {str(e)}")
            QMessageBox.critical(self, "对齐错误", f"处理失败: {e}")


    def show_context_menu(self, pos):
        """ 右键菜单：恢复了插入/删除，新增了合并子菜单 """
        menu = QMenu()
        item = self.table.itemAt(pos)
        current_row = item.row() if item else -1

        # --- 1. 恢复：插入行 (Insert) ---
        action_add = menu.addAction("插入行 (Insert)")
        # 这里直接调用你原本应该有的 insert_row 方法
        if hasattr(self, 'insert_row'):
            action_add.triggered.connect(self.insert_row)
        else:
            # 防御性代码：万一找不到方法，用默认逻辑
            action_add.triggered.connect(lambda: self.table.insertRow(current_row + 1 if current_row >= 0 else self.table.rowCount()))

        if item:
            # --- 2. 恢复：删除行 (Delete) ---
            action_del = menu.addAction("删除行 (Delete)")
            # 绑定到新的安全删除方法（后面会给）
            action_del.triggered.connect(lambda: self.delete_row_safe(current_row))
            
            menu.addSeparator()

            # --- 3. 新增：合并行 (子菜单) ---
            merge_menu = menu.addMenu("合并行 (Merge)")
            
            # 选项 A: 与上一行合并 (第一行时不可用)
            if current_row > 0:
                action_merge_up = merge_menu.addAction("与上一行合并 (Merge Up)")
                action_merge_up.triggered.connect(lambda: self.merge_row_up(current_row))
            
            # 选项 B: 与下一行合并 (最后一行时不可用)
            if current_row < self.table.rowCount() - 1:
                action_merge_down = merge_menu.addAction("与下一行合并 (Merge Down)")
                action_merge_down.triggered.connect(lambda: self.merge_row_down(current_row))

            menu.addSeparator()

            # --- 4. 原有功能：编辑/搜索 ---
            edit_action = menu.addAction("编辑 (Edit)")
            edit_action.triggered.connect(lambda: self.table.editItem(item))

            if item.column() > 0: # 排除ID列
                search_action = menu.addAction("Google Search")
                text = item.text()
                search_action.triggered.connect(lambda: self.open_google_search(text))

        menu.exec_(self.table.viewport().mapToGlobal(pos))


    # --- 新增的辅助方法 ---

    def delete_row_safe(self, row_idx):
        """ 删除行并刷新 ID (替代原有的删除逻辑) """
        self.table.removeRow(row_idx)
        self.refresh_row_ids()

    def merge_row_up(self, current_row):
        """ 将当前行合并到上一行：目标是 row-1, 来源是 row """
        target_row = current_row - 1
        self._merge_two_rows(target_row, current_row)

    def merge_row_down(self, current_row):
        """ 将下一行合并到当前行：目标是 row, 来源是 row+1 """
        target_row = current_row
        source_row = current_row + 1
        self._merge_two_rows(target_row, source_row)

    def _merge_two_rows(self, target_row, source_row):
        """ 核心合并逻辑：把 source_row 的内容拼接到 target_row 后，删除 source_row """
        # 1. 获取两行内容 (原文 Col 1, 译文 Col 2)
        s1 = self.table.item(target_row, 1).text() if self.table.item(target_row, 1) else ""
        t1 = self.table.item(target_row, 2).text() if self.table.item(target_row, 2) else ""
        
        s2 = self.table.item(source_row, 1).text() if self.table.item(source_row, 1) else ""
        t2 = self.table.item(source_row, 2).text() if self.table.item(source_row, 2) else ""

        # 2. 拼接 (使用换行符分隔，如果想要空格分隔改成 " ")
        new_s = (s1 + "\n" + s2).strip()
        new_t = (t1 + "\n" + t2).strip()

        # 3. 写回目标行
        self.table.setItem(target_row, 1, QTableWidgetItem(new_s))
        self.table.setItem(target_row, 2, QTableWidgetItem(new_t))

        # 4. 删除来源行
        self.table.removeRow(source_row)
        
        # 5. 刷新界面 (ID 重排，行高自适应)
        self.refresh_row_ids()
        self.table.resizeRowsToContents()


    def refresh_row_ids(self):
        """ 辅助方法：重新生成第一列的 ID 序号 """
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if not item:
                item = QTableWidgetItem()
                self.table.setItem(row, 0, item)
            item.setText(str(row + 1))



    def update_status_labels(self):
        s_count = len(self.source_subs_obj)
        t_count = len(self.target_subs_obj)
        
        info = []
        if s_count: info.append(f"原文: {s_count}行")
        if t_count: info.append(f"译文: {t_count}行")
        
        mode_idx = self.combo_mode.currentIndex()
        if mode_idx == 0: # Source Master
            self.lbl_status.setText(f"当前模式: [以原文为准] | " + " | ".join(info))
        else: # Target Master
            self.lbl_status.setText(f"当前模式: [以译文为准] | " + " | ".join(info))
            
        self.btn_action.setEnabled(s_count > 0 and t_count > 0)
        
        # 按钮文字变化
        if s_count > 0 and t_count > 0:
            if s_count != t_count:
                self.btn_action.setText("🛠️ 自动对齐并分析")
                self.btn_action.setStyleSheet("background-color: #f28b82; color: #202124;") # 红色提示
            else:
                self.btn_action.setText("🚀 直接分析")
                self.btn_action.setStyleSheet("background-color: #8ab4f8; color: #202124;")

    def update_preview(self):
        """简单的预览，不进行复杂处理"""
        # 显示原始文本，ASS清洗一下
        s_txt = [clean_ass_text(s.plaintext) for s in self.source_subs_obj]
        t_txt = [clean_ass_text(t.plaintext) for t in self.target_subs_obj]
        
        # 仅仅为了界面不空着，取最大行数显示
        rows = max(len(s_txt), len(t_txt))
        self.table.setRowCount(rows)
        for i in range(rows):
            self.table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            self.table.setItem(i, 1, QTableWidgetItem(s_txt[i] if i < len(s_txt) else ""))
            self.table.setItem(i, 2, QTableWidgetItem(t_txt[i] if i < len(t_txt) else ""))

    def handle_thread_error(self, err_msg):
        """ 专门处理线程抛出的致命错误 """
        self.progress.hide()
        self.btn_action.setEnabled(True)
        self.table.setEnabled(True)
        self.lbl_status.setText("发生错误")
        QMessageBox.critical(self, "运行错误", f"后台处理失败:\n{err_msg}\n\n请检查终端(Console)获取详细日志。")

    def run_process(self):
        """ 从表格读取内容并开始分析 """
        # 1. 基础检查
        key = self.api_input.text().strip()
        if not key: 
            return QMessageBox.warning(self, "缺少 API Key", "请输入 Gemini API Key。")
        self.settings.setValue("api_key", key)

        row_count = self.table.rowCount()
        if row_count == 0:
             return QMessageBox.warning(self, "空数据", "表格为空，请先加载文件。")

        # 2. 从表格 UI 抓取数据 (WYSIWYG: 所见即所得)
        source_lines = []
        target_lines = []
        
        valid_rows = 0
        for i in range(row_count):
            # 获取第1列(原文)和第2列(译文)
            it_s = self.table.item(i, 1)
            it_t = self.table.item(i, 2)
            
            txt_s = it_s.text().strip() if it_s else ""
            txt_t = it_t.text().strip() if it_t else ""
            
            # 即使有一边为空也可以提交(可能是漏译检查)，但全空则跳过
            source_lines.append(txt_s)
            target_lines.append(txt_t)
            valid_rows += 1
            
        logging.info(f"Starting LQA Check for {valid_rows} rows from table.")

        # 3. 锁定界面
        self.btn_action.setEnabled(False)
        self.table.setEnabled(False) # 分析时禁止修改，防止错位
        self.progress.show()
        self.full_results.clear()
        
        # 4. 启动 Worker
        self.start_lqa(key, source_lines, target_lines)



    def on_align_finished(self, aligned_list, mode):
        # aligned_list 是 "被修改的那一方" 的新文本列表
        # 我们需要根据模式组装 s_txt 和 t_txt
        
        if mode == 'source_master':
            # 原文是基准，译文被修改
            s_txt = [clean_ass_text(s.plaintext) for s in self.source_subs_obj]
            t_txt = aligned_list
            self.lbl_status.setText(f"对齐完成。原文 {len(s_txt)} 行 vs 对齐后译文 {len(t_txt)} 行")
        else:
            # 译文是基准，原文被修改
            t_txt = [clean_ass_text(t.plaintext) for t in self.target_subs_obj]
            s_txt = aligned_list
            self.lbl_status.setText(f"对齐完成。对齐后原文 {len(s_txt)} 行 vs 译文 {len(t_txt)} 行")
            
        # 刷新表格显示对齐结果
        self.table.setRowCount(len(s_txt))
        for i, (s, t) in enumerate(zip(s_txt, t_txt)):
            self.table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            self.table.setItem(i, 1, QTableWidgetItem(s)) # 原文列
            self.table.setItem(i, 2, QTableWidgetItem(t)) # 译文列
            # 标记一下哪一列是生成的
            if mode == 'target_master':
                self.table.item(i, 1).setForeground(QBrush(QColor("#f28b82"))) # 原文是生成的，标红
            else:
                self.table.item(i, 2).setForeground(QBrush(QColor("#f28b82"))) # 译文是生成的

        # 继续下一步
        key = self.settings.value("api_key")
        self.start_lqa(key, s_txt, t_txt)

    def start_lqa(self, key, source_lines, target_lines):
        # ... 之前的代码 ...
        self.worker = LQAWorker(key, source_lines, target_lines)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.batch_finished.connect(self.on_lqa_batch)
        self.worker.finished.connect(self.on_lqa_finished)
        
        # 连接错误处理（虽然我们在 run 里面 catch 住了，以防万一）
        # self.worker.error_occurred.connect(self.handle_thread_error) 
        
        self.worker.start()

    # ---------------------------------------------------------
    # 补全缺失的回调函数
    # ---------------------------------------------------------

    def update_progress(self, current, total, message):
        """ 接收 Worker 发来的进度信号 """
        self.progress.setMaximum(total)
        self.progress.setValue(current)
        self.lbl_status.setText(message)

    def on_lqa_batch(self, results):
        """ 接收 Worker 发来的一批审查结果，并填入表格 """
        logging.info(f"Received batch results: {len(results)} items")
        
        for res in results:
            # 获取行号 (我们在 Worker 里强制校准过 id 了)
            row_idx = res.get('id', -1)
            
            # 安全检查：防止 API 幻觉返回了不存在的行号
            if row_idx < 0 or row_idx >= self.table.rowCount():
                logging.warning(f"Skipping invalid row index: {row_idx}")
                continue

            # 1. 填入评分
            score = res.get('score', 0)
            item_score = QTableWidgetItem(str(score))
            item_score.setTextAlignment(Qt.AlignCenter)
            
            # 根据分数给背景色：红(差) -> 黄(中) -> 绿(好)
            if score < 60: 
                item_score.setBackground(QColor("#ffcccc")) 
            elif score < 80: 
                item_score.setBackground(QColor("#fff4cc")) 
            else: 
                item_score.setBackground(QColor("#ccffcc")) 
            
            self.table.setItem(row_idx, 3, item_score)

            # 2. 填入问题标签
            issues = res.get('issues', [])
            if isinstance(issues, list):
                issues_str = ", ".join(issues)
            else:
                issues_str = str(issues)
            self.table.setItem(row_idx, 4, QTableWidgetItem(issues_str))

            # 3. 填入修改建议
            suggestion = res.get('suggestion', '')
            self.table.setItem(row_idx, 5, QTableWidgetItem(str(suggestion)))
            
            # 强制刷新表格 UI，让用户看着它一行行填进去
            self.table.viewport().update()

    def on_lqa_finished(self):
        """ 审查全部完成 """
        logging.info("LQA Finished signal received.")
        self.progress.hide()
        self.btn_action.setEnabled(True)
        self.table.setEnabled(True)
        self.lbl_status.setText("审查完成！")
        QMessageBox.information(self, "完成", "所有字幕行审查完毕，请查看表格结果。")


    def on_lqa_batch(self, batch):
        for item in batch:
            if 'id' not in item: continue
            rid = item['id']
            if rid >= self.table.rowCount(): continue
            
            self.full_results[rid] = item
            
            # 分数
            score = item.get('score', 0)
            it = QTableWidgetItem(str(score))
            it.setTextAlignment(Qt.AlignCenter)
            
            # --- 颜色修改开始 ---
            if score == 10: 
                # 满分：深绿色 (Google Green)
                it.setForeground(QBrush(QColor("#188038"))) 
                it.setFont(QFont("Segoe UI", 9, QFont.Bold))
            elif score < 6: 
                # 不及格：深红色 (Google Red)
                it.setForeground(QBrush(QColor("#d93025")))
                it.setFont(QFont("Segoe UI", 9, QFont.Bold))
            else:
                # 普通分数：黑色
                it.setForeground(QBrush(QColor("#202124")))
            # --- 颜色修改结束 ---

            self.table.setItem(rid, 3, it)
            
            self.table.setItem(rid, 4, QTableWidgetItem(", ".join(item.get('issues', []))))
            self.table.setItem(rid, 5, QTableWidgetItem(item.get('suggestion', "")))
            self.table.scrollToItem(it)

    def on_all_done(self):
        self.progress.hide()
        self.table.setEnabled(True)
        self.btn_action.setEnabled(True)
        self.lbl_status.setText("✅ 分析完成")

    def on_row_clicked(self, item):
        r = item.row()
        if r not in self.full_results: return
        d = self.full_results[r]
        self.detail_text.setPlainText(f"""
[评分] {d.get('score')}
[类型] {d.get('issues')}
[建议] {d.get('suggestion')}

=== 点评 ===
{d.get('comment')}
""")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = LQAModernWindowV3()
    w.show()
    sys.exit(app.exec())
