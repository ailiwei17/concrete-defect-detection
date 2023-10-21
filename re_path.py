in_file = open("2007_train.txt", "r")
out_file = open("train_result.txt", "w")
lines = in_file.readlines()

for i in lines:
    i = i.replace("/home/mist", "..")
    out_file.write(i)

in_file.close()
out_file.close()

in_file = open("2007_val.txt", "r")
out_file = open("val_result.txt", "w")
lines = in_file.readlines()

for i in lines:
    i = i.replace("/home/mist", "..")
    out_file.write(i)

in_file.close()
out_file.close()
