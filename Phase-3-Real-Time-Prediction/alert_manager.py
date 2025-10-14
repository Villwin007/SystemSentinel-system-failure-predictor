import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json
from datetime import datetime
import os

class AlertManager:
    def __init__(self):
        self.alert_history = []
        
    def send_console_alert(self, message):
        """Send alert to console"""
        print("\n" + "="*60)
        print("🚨 ALERT TRIGGERED!")
        print("="*60)
        print(message)
        print("="*60 + "\n")
        
        self.log_alert(message)
    
    def send_email_alert(self, message, subject="System Anomaly Detected"):
        """Send alert via email (configure your SMTP settings)"""
        try:
            # Configure these settings for your email
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "your_email@gmail.com"
            sender_password = "your_app_password"
            receiver_email = "admin@yourcompany.com"
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            print("✅ Email alert sent")
            self.log_alert(f"Email sent: {subject}")
            
        except Exception as e:
            print(f"❌ Email alert failed: {e}")
    
    def send_slack_alert(self, message, webhook_url=None):
        """Send alert to Slack"""
        try:
            if webhook_url is None:
                print("⚠️  No Slack webhook URL configured")
                return
                
            slack_data = {'text': message}
            response = requests.post(
                webhook_url, 
                data=json.dumps(slack_data),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                print("✅ Slack alert sent")
            else:
                print(f"❌ Slack alert failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Slack alert failed: {e}")
    
    def log_alert(self, message):
        """Log alert to file"""
        alert_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        self.alert_history.append(alert_entry)
        
        # Save to file
        with open('alerts.log', 'a', encoding='utf-8') as f:
            f.write(f"{alert_entry['timestamp']} - {message}\n")
        
        # Keep only last 100 alerts in memory
        if len(self.alert_history) > 100:
            self.alert_history.pop(0)

# Global alert manager instance
alert_manager = AlertManager()

def send_alert(message, method='console', **kwargs):
    """
    Send alert using specified method
    """
    if method == 'console':
        alert_manager.send_console_alert(message)
    elif method == 'email':
        alert_manager.send_email_alert(message, **kwargs)
    elif method == 'slack':
        alert_manager.send_slack_alert(message, **kwargs)