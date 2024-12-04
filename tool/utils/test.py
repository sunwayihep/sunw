# import subprocess
# import time

# def test_run():
#     print("=== 使用 subprocess.run ===")
#     # run 会等待命令执行完成
#     result = subprocess.run(['ping', 'localhost', '-c', '3'], 
#                           capture_output=True,
#                           text=True)
#     print(f"命令执行完成，输出：\n{result.stdout}")

# def test_popen():
#     print("\n=== 使用 subprocess.Popen ===")
#     # Popen 立即返回进程对象
#     process = subprocess.Popen(['ping', 'localhost', '-c', '3'],
#                              stdout=subprocess.PIPE,
#                              text=True)
    
#     # 可以在进程执行期间做其他事情
#     print("进程已启动，但还在运行中...")
    
#     # 实时获取输出
#     # while True:
#     for output in process.stdout:
#         # output = process.stdout.readline()
#         if output == '' and process.poll() is not None:
#             break
#         if output:
#             print(f"实时输出: {output.strip()}")
            
#     # 最后获取所有剩余输出
#     stdout, stderr = process.communicate()
#     print("进程执行完成")

# if __name__ == '__main__':
#     # test_run()
#     test_popen()

if __name__ == '__main__':


    class Test:
        def __init__(self, a: int, b: int):
            self.a = a
            self.b = b

        def __str__(self):
            return f"a_{self.a}_b_{self.b}"
        
    test = Test(1, 2)
    print(f'test: {test}')
    import os
    print(os.path.join('a', str(test)))
    print(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
