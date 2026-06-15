import json
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'VoiceShield: Performance Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf():
    pdf = PDF()
    pdf.add_page()
    
    with open('../models/metrics_report.json', 'r') as f:
        metrics = json.load(f)
    
    val = metrics.get('validation', {})
    acc = val.get('accuracy', 0) * 100
    f1 = val.get('f1', 0) * 100
    eer = val.get('eer', 0) * 100
    
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Evaluation Metrics:', 0, 1)
    
    pdf.set_font('helvetica', '', 11)
    pdf.cell(0, 8, f'Accuracy: {acc:.2f}%', 0, 1)
    pdf.cell(0, 8, f'F1 Score: {f1:.2f}%', 0, 1)
    pdf.cell(0, 8, f'Equal Error Rate (EER): {eer:.2f}%', 0, 1)
    pdf.ln(5)
    
    per_class = val.get('per_class_accuracy', {})
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Per-Class Accuracy:', 0, 1)
    pdf.set_font('helvetica', '', 11)
    for cls, val_acc in per_class.items():
        pdf.cell(0, 8, f'{cls}: {val_acc * 100:.2f}%', 0, 1)
    
    pdf.ln(10)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Confusion Matrix:', 0, 1)
    pdf.image('../models/confusion_matrix.png', x=15, w=120)
    
    pdf.ln(10)
    pdf.cell(0, 10, 'ROC Curve:', 0, 1)
    pdf.image('../models/roc_curve.png', x=15, w=120)
    
    pdf.output('../models/performance_report.pdf')
    print("Performance report PDF generated successfully at ../models/performance_report.pdf")

if __name__ == '__main__':
    generate_pdf()
