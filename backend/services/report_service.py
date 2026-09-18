import json

class ReportService:
    def generate_json_report(self, prediction_result: dict) -> dict:
        return prediction_result
        
    def generate_html_report(self, prediction_result: dict) -> str:
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .disclaimer {{ font-size: 0.8em; color: #777; margin-top: 40px; border-top: 1px solid #ddd; padding-top: 10px; }}
                @media print {{
                    body {{ margin: 0; }}
                    .no-print {{ display: none; }}
                }}
            </style>
        </head>
        <body>
            <h1>InsureAI Prediction Report</h1>
            <div class="summary">
                <p><strong>Prediction:</strong> {prediction_result.get('prediction')}</p>
                <p><strong>Claim Probability:</strong> {prediction_result.get('claim_probability', 0):.2%}</p>
                <p><strong>Risk Score:</strong> {prediction_result.get('risk_score')}/100</p>
                <p><strong>Risk Category:</strong> {prediction_result.get('risk_category')}</p>
                <p><strong>Timestamp:</strong> {prediction_result.get('timestamp')}</p>
            </div>
            
            <h3>Recommendations</h3>
            <ul>
                {''.join(f'<li>{r}</li>' for r in prediction_result.get('recommendations', []))}
            </ul>
            
            <div class="disclaimer">
                {prediction_result.get('disclaimer')}
            </div>
        </body>
        </html>
        """
        return html
