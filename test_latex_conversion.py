import pypandoc
import os

# Define your LaTeX content
latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\title{Sample Document}
\author{Author Name}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}

This is a sample LaTeX document converted to Word using pypandoc.

\end{document}
"""

# Set the output Word file path (in the working directory)
docx_filename = "output.docx"
docx_file_path = os.path.join(os.getcwd(), docx_filename)

# Convert the LaTeX string directly to Word
pypandoc.convert_text(
    latex_content,
    to='docx',
    format='latex',
    outputfile=docx_file_path
)

print(f"Word document generated at: {docx_file_path}")
