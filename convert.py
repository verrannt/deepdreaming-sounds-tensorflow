import scipy.io.wavfile as wv

class SampleRate():
    #fileNames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    def fortyFourKiloHertz(directory,filename):
        file = wv.read(directory+filename)
        if file[0] != 44100:
            wv.write(directory+filename, 44100, file[1])
        else:
            pass
