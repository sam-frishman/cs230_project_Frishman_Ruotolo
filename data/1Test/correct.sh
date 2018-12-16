#!/bin/bash
mv test_signs/* right/
cd others
shuf -n 1 -e * | xargs -i mv {} ../test_signs/
cd ..
