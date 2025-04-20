#!/bin/bash
cd serve || exit
find src/ | entr -r elm make src/Main.elm --output compiled/Main.js
