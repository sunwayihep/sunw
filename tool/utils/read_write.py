
def read (file_name) :
  ret = []
  with open (file_name, 'r') as f:
    for line in f:
      ret.append(float(line))
  return ret

def write (file_name, content) :
  with open (file_name, 'w') as f:
    for line in content:
      f.write(str(line))
      f.write('\n')

def read_and_write (input_file, output_file) :
  ret = read(input_file)
  content = list(set(ret))
  content.sort()
  write(output_file, content)

