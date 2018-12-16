#!/bin/bash
mv test_signs/* wrong/
cd others
shuf -n 1 -e * | xargs -i mv {} ../test_signs/
cd ..
