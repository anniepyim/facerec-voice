#!/usr/bin/env bash

./../ngrok/ngrok http -auth="username:password" 8082 &
(cd ../MagicMirror && npm run start .) &
python textandtalk_http.py &
python init_web.py && fg