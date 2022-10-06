n = 1
while True:

    try:
        n += 1
        print(x)
    except Exception:
        print("Something broke!")
        if n >= 20:
            break
# Something broke!