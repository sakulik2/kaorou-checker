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

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QTableWidget, QTableWidgetItem, 
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
        self.setWindowTitle("LQA Pro - ASS支持 & 双向对齐")
        self.resize(1600, 1000)
        self.settings = QSettings("LQA_Pro_v3")
        
        self.source_subs_obj = [] 
        self.target_subs_obj = []
        self.source_texts_display = [] # 最终显示在表格里的原文
        self.target_texts_display = [] # 最终显示在表格里的译文
        self.full_results = {}

        self.setup_ui()
        self.apply_styles()

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
        self.combo_mode.currentIndexChanged.connect(self.update_status_labels)

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


    def load_file(self, ftype):
        # 1. 在过滤器中加回 *.txt
        path, _ = QFileDialog.getOpenFileName(self, "选择字幕", "", "Subtitle (*.srt *.ass *.vtt *.txt)")
        if not path: return
        
        try:
            # 2. 专门处理 txt 文件
            if path.lower().endswith(".txt"):
                subs = pysubs2.SSAFile()
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # 创建一个伪造的字幕行
                            # start=0, end=360000000 (100小时)
                            # 这样做是为了保证如果在"对齐模式"下，这些文本永远在时间窗口内
                            ev = pysubs2.SSAEvent(start=0, end=360000000, text=line)
                            subs.events.append(ev)
            else:
                # 其他格式照旧用库加载
                subs = pysubs2.load(path)

            if ftype == 'source': self.source_subs_obj = subs
            else: self.target_subs_obj = subs
            
            self.update_status_labels()
            self.update_preview()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"文件解析失败: {e}")


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
        # 1. 基础检查
        key = self.api_input.text().strip()
        if not key: 
            return QMessageBox.warning(self, "缺少 API Key", "请输入 Gemini API Key。")
        self.settings.setValue("api_key", key)

        if not self.source_subs_obj or not self.target_subs_obj:
            return QMessageBox.warning(self, "缺少文件", "请先加载原文和译文文件！")

        s_count = len(self.source_subs_obj)
        t_count = len(self.target_subs_obj)
        
        logging.info(f"Run Process Triggered. Source Lines: {s_count}, Target Lines: {t_count}")

        if s_count == 0 or t_count == 0:
             return QMessageBox.warning(self, "空文件", "加载的文件似乎是空的，或者没有有效的字幕行。")

        # 2. 准备数据
        s_txt_raw = [clean_ass_text(s.plaintext) for s in self.source_subs_obj]
        t_txt_raw = [clean_ass_text(t.plaintext) for t in self.target_subs_obj]

        self.btn_action.setEnabled(False)
        self.table.setEnabled(False)
        self.progress.show()
        self.full_results.clear()

        # 3. 逻辑分流
        if s_count == t_count:
            logging.info("Line counts match. Starting LQA directly.")
            self.lbl_status.setText("行数一致，直接审查...")
            
            # 填充表格用于预览
            self.table.setRowCount(s_count)
            for i in range(s_count):
                self.table.setItem(i, 0, QTableWidgetItem(str(i+1)))
                self.table.setItem(i, 1, QTableWidgetItem(s_txt_raw[i]))
                self.table.setItem(i, 2, QTableWidgetItem(t_txt_raw[i]))
                
            self.start_lqa(key, s_txt_raw, t_txt_raw)
        else:
            logging.info(f"Line counts mismatch ({s_count} vs {t_count}). Starting AutoAligner.")
            self.lbl_status.setText(f"行数不符，开始对齐...")
            mode_idx = self.combo_mode.currentIndex()
            mode_str = 'source_master' if mode_idx == 0 else 'target_master'
            
            self.aligner = AutoAligner(key, self.source_subs_obj, self.target_subs_obj, mode_str)
            
            # 连接信号
            self.aligner.progress_update.connect(lambda c, t, s: self.lbl_status.setText(s))
            self.aligner.finished.connect(lambda res: self.on_align_finished(res, mode_str))
            
            # 关键：连接错误信号
            self.aligner.error_occurred.connect(self.handle_thread_error)
            
            self.aligner.start()



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
