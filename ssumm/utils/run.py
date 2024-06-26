import subprocess

def run(name='ka',useNE=0,NEmodel='',uselazy=0,useNE2=0):
    for k in range(1,10):
        k/=10
        subprocess.Popen(["start", "cmd", "/k", "python", "SSumM.py",f"--data={name}",f"--k={k}",f"--useNE={useNE}",f"--NEmodel={NEmodel}",f"--uselazy={uselazy}",f"--useNE2={useNE2}"], shell=True,cwd="E:\Desktop\PythonProjects\SSumMpy")

run(name='ka',useNE=1,NEmodel='gcncd',uselazy=0,useNE2=0)



"""










# 执行1.py
process1 = subprocess.Popen(["python", "1.py"], stdout=subprocess.PIPE)
output1, error1 = process1.communicate()
print("Output of 1.py:", output1.decode())

# 执行2.py
process2 = subprocess.Popen(["python", "2.py"], stdout=subprocess.PIPE)
output2, error2 = process2.communicate()
print("Output of 2.py:", output2.decode())
"""
