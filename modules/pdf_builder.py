from fpdf import FPDF

class ExamPaperPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(False, 15)
        self.first_page = True
        self.in_unit = False  
        self.w_qno = 12
        self.w_q = 118
        self.w_co = 15
        self.w_po = 15
        self.w_m = 15

  
    def header(self):
        if not self.first_page:
            self.set_font("helvetica", "", 8)
            self.cell(0, 6, f"Page {self.page_no()}", ln=True, align="R")
            self.ln(4)  

   
    def clean(self, text):
        if not text:
            return ""
        repl = {
            "–": "-", "—": "-", "•": "-",
            "“": '"', "”": '"', "‘": "'", "’": "'",
            "→": "->"
        }
        for k, v in repl.items():
            text = text.replace(k, v)
        return text.encode("latin-1", "ignore").decode("latin-1")

   
    def draw_first_page_header(self, branch, code, course, sem):
        self.set_font("helvetica", "B", 15)
        self.cell(0, 10,
                  "B.M.S. COLLEGE OF ENGINEERING, BENGALURU - 560019",
                  ln=True, align="C")

        self.set_font("helvetica", "B", 11)
        self.cell(0, 8, "Autonomous Institute Affiliated to VTU",
                  ln=True, align="C")
        self.ln(4)

        self.set_font("helvetica", "B", 13)
        self.cell(0, 9, "SEMESTER END MAIN EXAMINATIONS",
                  ln=True, align="C")
        self.ln(6)

        self.set_font("helvetica", "B", 10)
        self.cell(95, 8, f"Programme: B.E.    Semester: {sem}", 1)
        self.cell(0, 8, "Duration: 3 Hours", 1, ln=True)
        self.cell(95, 8, f"Branch: {branch}", 1)
        self.cell(0, 8, "Max Marks: 100", 1, ln=True)
        self.cell(95, 8, f"Course Code: {code}", 1)
        self.cell(0, 8, f"Course: {course}", 1, ln=True)

        self.ln(6)
        self.set_font("helvetica", "", 9)
        self.multi_cell(
            0, 6,
            "Instructions:\n"
            "1. Answer any TEN full questions, choosing two from each unit.\n"
            "2. Missing data may be suitably assumed."
        )
        self.ln(4)
        self.first_page = False

   
    def _draw_table_header(self):
        self.set_font("helvetica", "B", 9)
        self.cell(self.w_qno, 8, "Q.No", 1)
        self.cell(self.w_q, 8, "Questions", 1)
        self.cell(self.w_co, 8, "CO", 1, align="C")
        self.cell(self.w_po, 8, "PO", 1, align="C")
        self.cell(self.w_m, 8, "Marks", 1, ln=True, align="C")

   
    def unit_header(self, title):
        if self.get_y() > self.page_break_trigger - 40:
            self.add_page()

        self.in_unit = True
        self.set_font("helvetica", "B", 11)
        self.cell(0, 9, title, ln=True)
        self._draw_table_header()

    
    def _check_page(self, h):
        if self.get_y() + h > self.page_break_trigger:
            self.add_page()
            if self.in_unit:
                self._draw_table_header()  

    
    def _row_height(self, text):
        lines = self.multi_cell(self.w_q, 6, text, split_only=True)
        return 6 * len(lines)

    def question_row(self, qno, text, co, po, marks):
        text = self.clean(text)
        h = self._row_height(text)
        self._check_page(h)

        x, y = self.get_x(), self.get_y()
        self.multi_cell(self.w_qno, h, qno, 1, align="C")
        self.set_xy(x + self.w_qno, y)

        self.multi_cell(self.w_q, 6, text, 1)
        self.set_xy(x + self.w_qno + self.w_q, y)

        self.cell(self.w_co, h, co, 1, align="C")
        self.cell(self.w_po, h, po, 1, align="C")
        self.cell(self.w_m, h, marks, 1, ln=True, align="C")

   
    def or_row(self):
        self._check_page(10)
        self.set_font("helvetica", "B", 9)
        self.cell(
            self.w_qno + self.w_q + self.w_co + self.w_po + self.w_m,
            8,
            "OR",
            1,
            ln=True,
            align="C"
        )
