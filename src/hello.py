import os

#results_path = "/Volumes/pelvis/projects/evgenia/experiments"
results_path = "/mnt/netcache/pelvis/projects/evgenia/experiments"

def hello():
    os.system('ls /mnt/netcache/pelvis/projects/evgenia/')
    print(results_path)
    print("Hello SOL")
    file_path = os.path.join(results_path, "test1.txt")
    file = open(file_path, "w+")
    file.write("Some text")
    file.close()
    print("file written")
    f = open(os.path.join(results_path, "test.txt"), "r")
    c = f.read()
    f.close()
    print(c)





if __name__ == '__main__':
    hello()
