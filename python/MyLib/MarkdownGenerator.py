import os
from datetime import datetime
from typing import List, Dict, Any, Union, Optional

class MarkdownGenerator:
    """
    Markdown文件生成器类
    用于创建和编辑Markdown文档
    """
    
    def __init__(self, title: str = "", author: str = "", date_format: str = "%Y-%m-%d"):
        """
        初始化Markdown生成器
        
        参数:
        title: 文档标题
        author: 作者
        date_format: 日期格式
        """
        self.content = []
        self.title = title
        self.author = author
        self.date_format = date_format
        self.table_of_contents = []
        
        # 如果提供了标题，自动添加
        if title:
            self.add_title(title, level=1)
            if author:
                self.add_metadata("author", author)
            self.add_metadata("date", datetime.now().strftime(date_format))
            self.add_horizontal_rule()
    
    def add_title(self, text: str, level: int = 1) -> 'MarkdownGenerator':
        """
        添加标题
        
        参数:
        text: 标题文本
        level: 标题级别 (1-6)
        
        返回:
        自身实例，支持链式调用
        """
        if level < 1:
            level = 1
        elif level > 6:
            level = 6
        
        title_line = "#" * level + " " + text
        self.content.append(title_line)
        
        # 添加到目录（1-3级标题）
        if level <= 3:
            indent = "  " * (level - 1)
            self.table_of_contents.append(f"{indent}- [{text}](#{self._slugify(text)})")
        
        return self
    
    def add_paragraph(self, text: str) -> 'MarkdownGenerator':
        """
        添加段落
        
        参数:
        text: 段落文本
        
        返回:
        自身实例，支持链式调用
        """
        self.content.append(text)
        self.content.append("")  # 空行分隔
        return self
    
    def add_list(self, items: List[str], ordered: bool = False, indent_level: int = 0) -> 'MarkdownGenerator':
        """
        添加列表
        
        参数:
        items: 列表项
        ordered: 是否有序列表
        indent_level: 缩进级别
        
        返回:
        自身实例，支持链式调用
        """
        indent = "  " * indent_level
        
        for i, item in enumerate(items):
            if ordered:
                prefix = f"{indent}{i+1}. "
            else:
                prefix = f"{indent}- "
            self.content.append(f"{prefix}{item}")
        
        self.content.append("")  # 空行分隔
        return self
    
    def add_table(self, headers: List[str], rows: List[List[str]], 
                  align: List[str] = None, caption: str = "") -> 'MarkdownGenerator':
        """
        添加表格
        
        参数:
        headers: 表头列表
        rows: 数据行列表
        align: 对齐方式列表 ('left', 'center', 'right')
        caption: 表格标题
        
        返回:
        自身实例，支持链式调用
        """
        if align is None:
            align = ['left'] * len(headers)
        
        # 验证参数
        if len(headers) != len(align):
            raise ValueError("headers和align长度必须相同")
        
        for row in rows:
            if len(row) != len(headers):
                raise ValueError("每行的列数必须与表头相同")
        
        # 添加表格标题
        if caption:
            self.content.append(f"**{caption}**")
            self.content.append("")
        
        # 表头
        header_line = "| " + " | ".join(headers) + " |"
        self.content.append(header_line)
        
        # 分隔线
        align_chars = []
        for a in align:
            if a == 'left':
                align_chars.append(":---")
            elif a == 'center':
                align_chars.append(":---:")
            elif a == 'right':
                align_chars.append("---:")
            else:
                align_chars.append("---")
        
        separator_line = "| " + " | ".join(align_chars) + " |"
        self.content.append(separator_line)
        
        # 数据行
        for row in rows:
            row_line = "| " + " | ".join(row) + " |"
            self.content.append(row_line)
        
        self.content.append("")  # 空行分隔
        return self
    
    def add_code_block(self, code: str, language: str = "") -> 'MarkdownGenerator':
        """
        添加代码块
        
        参数:
        code: 代码文本
        language: 编程语言
        
        返回:
        自身实例，支持链式调用
        """
        self.content.append(f"```{language}")
        self.content.append(code)
        self.content.append("```")
        self.content.append("")  # 空行分隔
        return self
    
    def add_blockquote(self, text: str) -> 'MarkdownGenerator':
        """
        添加引用块
        
        参数:
        text: 引用文本
        
        返回:
        自身实例，支持链式调用
        """
        lines = text.split('\n')
        for line in lines:
            self.content.append(f"> {line}")
        self.content.append("")  # 空行分隔
        return self
    
    def add_horizontal_rule(self) -> 'MarkdownGenerator':
        """
        添加水平分割线
        
        返回:
        自身实例，支持链式调用
        """
        self.content.append("---")
        self.content.append("")  # 空行分隔
        return self
    
    def add_link(self, text: str, url: str, title: str = "") -> 'MarkdownGenerator':
        """
        添加链接
        
        参数:
        text: 链接文本
        url: 链接地址
        title: 链接标题（可选）
        
        返回:
        自身实例，支持链式调用
        """
        if title:
            link = f'[{text}]({url} "{title}")'
        else:
            link = f'[{text}]({url})'
        
        self.content.append(link)
        self.content.append("")  # 空行分隔
        return self
    
    def add_image(self, alt_text: str, url: str, title: str = "") -> 'MarkdownGenerator':
        """
        添加图片
        
        参数:
        alt_text: 替代文本
        url: 图片地址
        title: 图片标题（可选）
        
        返回:
        自身实例，支持链式调用
        """
        if title:
            image = f'![{alt_text}]({url} "{title}")'
        else:
            image = f'![{alt_text}]({url})'
        
        self.content.append(image)
        self.content.append("")  # 空行分隔
        return self
    
    def add_metadata(self, key: str, value: str) -> 'MarkdownGenerator':
        """
        添加元数据（YAML front matter）
        
        参数:
        key: 元数据键
        value: 元数据值
        
        返回:
        自身实例，支持链式调用
        """
        # 如果还没有YAML front matter，添加一个
        if not self.content or not self.content[0].startswith("---"):
            self.content.insert(0, "---")
            self.content.insert(1, f"{key}: {value}")
            self.content.insert(2, "---")
            self.content.insert(3, "")  # 空行
        else:
            # 找到YAML结束位置
            end_index = self.content.index("---", 1)
            self.content.insert(end_index, f"{key}: {value}")
        
        return self
    
    def add_table_of_contents(self, max_level: int = 3) -> 'MarkdownGenerator':
        """
        添加目录
        
        参数:
        max_level: 最大标题级别
        
        返回:
        自身实例，支持链式调用
        """
        if self.table_of_contents:
            self.add_title("目录", level=2)
            for item in self.table_of_contents:
                # 只显示指定级别以下的标题
                indent_level = (len(item) - len(item.lstrip())) // 2
                if indent_level < max_level:
                    self.content.append(item)
            self.content.append("")  # 空行分隔
            self.add_horizontal_rule()
        
        return self
    
    def add_custom_html(self, html: str) -> 'MarkdownGenerator':
        """
        添加自定义HTML
        
        参数:
        html: HTML代码
        
        返回:
        自身实例，支持链式调用
        """
        self.content.append(html)
        self.content.append("")  # 空行分隔
        return self
    
    def add_footnote(self, ref: str, text: str) -> 'MarkdownGenerator':
        """
        添加脚注
        
        参数:
        ref: 脚注引用标识
        text: 脚注文本
        
        返回:
        自身实例，支持链式调用
        """
        self.content.append(f"[^{ref}]: {text}")
        return self
    
    def add_footnote_reference(self, ref: str) -> str:
        """
        添加脚注引用
        
        参数:
        ref: 脚注引用标识
        
        返回:
        脚注引用标记
        """
        return f"[^{ref}]"
    
    def _slugify(self, text: str) -> str:
        """
        将文本转换为URL友好的slug
        
        参数:
        text: 原始文本
        
        返回:
        slug字符串
        """
        # 转换为小写，替换空格为连字符，移除特殊字符
        slug = text.lower()
        slug = slug.replace(' ', '-')
        slug = ''.join(c for c in slug if c.isalnum() or c == '-')
        return slug
    
    def clear(self) -> 'MarkdownGenerator':
        """
        清空内容
        
        返回:
        自身实例，支持链式调用
        """
        self.content = []
        self.table_of_contents = []
        return self
    
    def get_content(self) -> str:
        """
        获取生成的Markdown内容
        
        返回:
        Markdown文本
        """
        return "\n".join(self.content)
    
    def save(self, filepath: str, encoding: str = 'utf-8') -> bool:
        """
        保存到文件
        
        参数:
        filepath: 文件路径
        encoding: 文件编码
        
        返回:
        是否保存成功
        """
        try:
            with open(filepath, 'w', encoding=encoding) as f:
                f.write(self.get_content())
            return True
        except Exception as e:
            print(f"保存文件时出错: {e}")
            return False
    
    def load(self, filepath: str, encoding: str = 'utf-8') -> bool:
        """
        从文件加载
        
        参数:
        filepath: 文件路径
        encoding: 文件编码
        
        返回:
        是否加载成功
        """
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                self.content = f.read().splitlines()
            
            # 重新解析目录
            self.table_of_contents = []
            for line in self.content:
                if line.startswith('#') and ' ' in line:
                    level = len(line) - len(line.lstrip('#'))
                    text = line[level:].strip()
                    if level <= 3:
                        indent = "  " * (level - 1)
                        self.table_of_contents.append(f"{indent}- [{text}](#{self._slugify(text)})")
            
            return True
        except Exception as e:
            print(f"加载文件时出错: {e}")
            return False
    
    def __str__(self) -> str:
        """
        字符串表示
        
        返回:
        Markdown文本
        """
        return self.get_content()


# 高级功能扩展
class AdvancedMarkdownGenerator(MarkdownGenerator):
    """
    高级Markdown生成器
    扩展了更多功能
    """
    
    def add_checklist(self, items: List[Dict[str, Any]]) -> 'AdvancedMarkdownGenerator':
        """
        添加复选框列表
        
        参数:
        items: 列表项，每个项是字典 {'text': '...', 'checked': True/False}
        
        返回:
        自身实例，支持链式调用
        """
        for item in items:
            checkbox = "[x]" if item.get('checked', False) else "[ ]"
            self.content.append(f"- {checkbox} {item['text']}")
        
        self.content.append("")  # 空行分隔
        return self
    
    def add_definition_list(self, definitions: Dict[str, str]) -> 'AdvancedMarkdownGenerator':
        """
        添加定义列表
        
        参数:
        definitions: 定义字典 {术语: 定义}
        
        返回:
        自身实例，支持链式调用
        """
        for term, definition in definitions.items():
            self.content.append(f"{term}")
            self.content.append(f": {definition}")
            self.content.append("")  # 空行
        
        return self
    
    def add_math_equation(self, equation: str, inline: bool = False) -> 'AdvancedMarkdownGenerator':
        """
        添加数学公式
        
        参数:
        equation: 数学公式
        inline: 是否行内公式
        
        返回:
        自身实例，支持链式调用
        """
        if inline:
            self.content.append(f"${equation}$")
        else:
            self.content.append(f"$${equation}$$")
        
        self.content.append("")  # 空行分隔
        return self
    
    def add_mermaid_diagram(self, diagram_code: str, title: str = "") -> 'AdvancedMarkdownGenerator':
        """
        添加Mermaid图表
        
        参数:
        diagram_code: Mermaid代码
        title: 图表标题
        
        返回:
        自身实例，支持链式调用
        """
        if title:
            self.content.append(f"**{title}**")
            self.content.append("")
        
        self.content.append("```mermaid")
        self.content.append(diagram_code)
        self.content.append("```")
        self.content.append("")  # 空行分隔
        return self
    
    def add_collapsible_section(self, title: str, content: str) -> 'AdvancedMarkdownGenerator':
        """
        添加可折叠部分
        
        参数:
        title: 部分标题
        content: 折叠内容
        
        返回:
        自身实例，支持链式调用
        """
        self.content.append(f"<details>")
        self.content.append(f"<summary>{title}</summary>")
        self.content.append("")
        self.content.append(content)
        self.content.append("")
        self.content.append(f"</details>")
        self.content.append("")  # 空行分隔
        return self


# 使用示例
def excample_usage():
    # 创建基本Markdown文档
    md = MarkdownGenerator(title="Python数据分析报告", author="数据分析师")
    
    # 添加目录
    md.add_table_of_contents()
    
    # 添加章节
    md.add_title("项目概述", level=2)
    md.add_paragraph("这是一个关于Python数据分析的示例报告。")
    
    # 添加列表
    md.add_title("主要功能", level=3)
    md.add_list([
        "数据清洗和预处理",
        "统计分析和可视化",
        "机器学习建模",
        "结果报告生成"
    ])
    
    # 添加表格
    md.add_title("数据统计", level=2)
    md.add_table(
        headers=["指标", "数值", "单位"],
        rows=[
            ["样本数量", "1000", "个"],
            ["平均值", "25.4", "cm"],
            ["标准差", "3.2", "cm"],
            ["最大值", "35.1", "cm"],
            ["最小值", "15.8", "cm"]
        ],
        align=['left', 'right', 'center'],
        caption="数据统计摘要"
    )
    
    # 添加代码块
    md.add_title("示例代码", level=2)
    md.add_code_block("""import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('data.csv')

# 数据分析
summary = data.describe()
print(summary)""", language="python")
    
    # 添加引用
    md.add_blockquote("数据是新时代的石油，而分析是炼油厂。")
    
    # 添加链接
    md.add_title("参考资料", level=2)
    md.add_link("Python官方文档", "https://docs.python.org/3/", "Python编程语言官方文档")
    
    # 保存文件
    md.save("report.md")
    print("Markdown文件已生成: report.md")
    
    # 显示内容
    print("\n生成的Markdown内容预览:")
    print("=" * 50)
    print(md.get_content()[:500] + "...")
    
    # 高级功能示例
    print("\n" + "=" * 50)
    print("高级功能示例:")
    
    advanced_md = AdvancedMarkdownGenerator(title="高级报告")
    
    # 添加复选框列表
    advanced_md.add_checklist([
        {"text": "完成数据清洗", "checked": True},
        {"text": "进行统计分析", "checked": True},
        {"text": "创建可视化图表", "checked": False},
        {"text": "撰写最终报告", "checked": False}
    ])
    
    # 添加定义列表
    advanced_md.add_definition_list({
        "API": "应用程序编程接口",
        "JSON": "JavaScript对象表示法",
        "REST": "表述性状态传递"
    })
    
    # 添加数学公式
    advanced_md.add_math_equation("E = mc^2")
    advanced_md.add_math_equation("\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}")
    
    # 添加Mermaid图表
    mermaid_code = """graph TD
    A[开始] --> B[数据收集]
    B --> C[数据清洗]
    C --> D[数据分析]
    D --> E[结果可视化]
    E --> F[报告生成]
    F --> G[结束]"""
    
    advanced_md.add_mermaid_diagram(mermaid_code, "数据分析流程")
    
    # 保存高级文档
    advanced_md.save("advanced_report.md")
    print("高级Markdown文件已生成: advanced_report.md")