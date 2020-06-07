#!/usr/bin/env bash

kill $(ps aux | grep 'python' | awk '{print $2}')
kill $(ps aux | grep 'npm' | awk '{print $2}')
kill $(ps aux | grep 'startup' | awk '{print $2}')
kill $(ps aux | grep 'ngrok' | awk '{print $2}')