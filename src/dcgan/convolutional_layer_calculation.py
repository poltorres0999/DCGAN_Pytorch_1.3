w = 95
k = 5
s = 2
p = 1

n = 5
k1 = 5
s1 = 2
p1 = 1
n_layers = 6


def conv_output(w, k , s, p):
    return ((w - k + 2 * p) / s) + 1

def transposed_conv_output(n, k, s, p):
    return s * (n - 1) + k - 2*p

def show_conv_output_size(w, k, s, p, n_layers):
    current = w
    print(f"Layer_0 -> Output size: {current}")
    for i in range(n_layers - 1):
        current = conv_output(current, k, s, p)
        print(f"Layer_{i + 1} -> Output size: {current}")

def show_conv_trans_output_size(n, k, s, p, n_layers):
    current = n
    print(f"Layer_0 -> Output size: {current}")
    for i in range(n_layers -1):
        current = transposed_conv_output(current, k, s, p)
        print(f"Layer_{i+1} -> Output size: {current}")

print(f" ##### Convolutional transposed generator model #####")
show_conv_trans_output_size(n, k1, s1, p1, n_layers)
print(f" ##### Convolutional discriminator model #####")
show_conv_output_size(w, k, s, p, n_layers)


