#!/usr/bin/env bash

kill $(ps aux | grep 'python' | awk '{print $2}')
kill $(ps aux | grep 'npm' | awk '{print $2}')
kill $(ps aux | grep 'startup' | awk '{print $2}')
kill $(ps aux | grep 'ngrok' | awk '{print $2}')

kill $(sudo netstat -ap | grep :8081 | awk '{print $7}' | awk -F / '{print $1}')
kill $(sudo netstat -ap | grep :8082 | awk '{print $7}' | awk -F / '{print $1}')
kill $(sudo netstat -ap | grep :8083 | awk '{print $7}' | awk -F / '{print $1}')