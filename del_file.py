import os

def delete_large_files(folder_path, size_limit_mb=100):
    """
    删除指定文件夹中大于指定大小的文件。
    
    :param folder_path: 要扫描的文件夹路径
    :param size_limit_mb: 文件大小限制（单位：MB，默认为 100MB）
    """
    size_limit_bytes = size_limit_mb * 1024 * 1024  # 将 MB 转换为字节
    deleted_files = []

    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                if file_size > size_limit_bytes:
                    print(f"删除文件：{file_path}，大小：{file_size / (1024 * 1024):.2f} MB")
                    os.remove(file_path)  # 删除文件
                    deleted_files.append(file_path)
            except Exception as e:
                print(f"处理文件 {file_path} 时出错：{e}")

    print("\n删除完成！")
    if deleted_files:
        print(f"共删除 {len(deleted_files)} 个文件：")
        for file in deleted_files:
            print(file)
    else:
        print("没有找到大于 100MB 的文件。")

# 使用示例
if __name__ == "__main__":
    folder_path = input("请输入要扫描的文件夹路径：")
    delete_large_files(folder_path)