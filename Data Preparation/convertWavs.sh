mkdir convertedWavs
find -name "*.wav" -exec bash -c \
'ffmpeg -i "{}" -y -ar 8000 -ac 1 -acodec pcm_s16le \
"convertedWavs/${0/.wav}-c.wav"' {} \;
