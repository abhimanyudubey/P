import smtplib
import os
import sys
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email.Utils import COMMASPACE, formatdate
from email import Encoders
from optparse import OptionParser

parser = OptionParser()

parser.add_option(
    "-x", "--server", 
    dest    = "server",
    default = "smtp.gmail.com",
    help    = "specify SMTP server URL",
    metavar = "URL")

parser.add_option(
    "-v", "--verbose", 
    dest    = "verb",
    action  = "count",
    help    = "increase verbosity level")

parser.add_option(
    "-r", "--port", 
    dest    = "port",
    default = 587,
    type    = "int",
    help    = "specify SMTP server port",
    metavar = "PORT")

parser.add_option(
    "-p", "--password", 
    dest    = "password",
    default = "",
    help    = "specify SMTP password",
    metavar = "PASSWD")

parser.add_option(
    "-u", "--username", 
    dest    = "username",
    default = "",
    help    = "specify SMTP user name", 
    metavar = "NAME")

parser.add_option(
    "-f", "--from", 
    dest    = "send_from",
    default = "",
    help    = "specify the FROM field")

parser.add_option(
    "-t", "--to", 
    dest    = "send_to",
    default = "",
    help    = "specify the TO field")

parser.add_option(
    "-b", "--body", 
    dest    = "body",
    default = "",
    help    = "specify the body",
    metavar = "TXT")

parser.add_option(
    "-s", "--subject", 
    dest    = "subject",
    default = "",
    help    = "specify the subject")


# --------------------------------------------------------------------
def make_msg(send_from,
             send_to,
             subject,
             body,
             files):
# --------------------------------------------------------------------
    assert type(send_to)==list
    assert type(files)==list

    msg            = MIMEMultipart()
    msg['From']    = send_from
    msg['To']      = COMMASPACE.join(send_to)
    msg['Date']    = formatdate(localtime=True)
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body))
    
    for f in files:
        part = MIMEBase('application', "octet-stream")
        part.set_payload(open(f,"rb").read())
        Encoders.encode_base64(part)
        part.add_header('Content-Disposition', 
                        'attachment; filename="%s"' % os.path.basename(f))
        msg.attach(part)

    if options.verb:
        print msg

    return msg

# --------------------------------------------------------------------
def send_tsl(msg,
             server,
             port,
             username,
             password):
# --------------------------------------------------------------------
    smtp = smtplib.SMTP(server, port)
    smtp.ehlo()
    smtp.starttls()
    smtp.ehlo()
    smtp.login(username, password)
    smtp.sendmail(msg['From'], msg['To'], msg.as_string())
    smtp.close()

# --------------------------------------------------------------------
if __name__ == "__main__":
# --------------------------------------------------------------------

    (options, args) = parser.parse_args()
    
    msg = make_msg(options.send_from,
                   [options.send_to],
                   options.subject,
                   options.body,
                   args)

    send_tsl(msg,
             options.server,
             options.port,
             options.username,
             options.password) ;

    sys.exit(0)
