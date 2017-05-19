mkdir unprocessed
mv *.wav unprocessed

cd unprocessed
find -name "*.wav" -exec bash -c \
'ffmpeg -i "{}" -y -ar 8000 -ac 1 -acodec pcm_s16le \
"../${0/.wav}-p.wav"' {} \;
cd ..

mkdir processed
mv *.wav processed

