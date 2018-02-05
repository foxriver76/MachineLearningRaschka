# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: moritz
"""

from flask import Flask, render_template

app = Flask(__name__)

"""App Route legt fest wo Index zu sehen sein soll"""
@app.route('/')
def index():
    return render_template('first_app.html')

if __name__ == '__main__':
    app.run()