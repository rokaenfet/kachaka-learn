with open("t.txt", "r") as f:
    a = [n.strip().replace("\n","").replace("(","").replace(")","").replace(" ","_").upper()+f"={i}\n" for i,n in enumerate(f.readlines())]
with open("t.txt","w") as f:
    f.write("".join(a))