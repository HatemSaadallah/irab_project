#!/bin/bash
#to count words in sub directories
find . -type f -exec cat {} + | wc -w


