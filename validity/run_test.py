import os
import re
import subprocess
import shutil
import sys

# --- 路径配置 ---
# 脚本位于 validity/test.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 原版二进制所在目录 (build-clang-o2/src)
ORIG_SRC_DIR = os.path.join(BASE_DIR, "coreutils-9.8/build-clang-o2/src")
# 测试运行根目录
BUILD_DIR = os.path.join(BASE_DIR, "coreutils-9.8/build-clang-o2")
# 混淆版根目录 (父目录下的 dataset)
OBF_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../dataset"))

class SABREScaleValidator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.scales = ["small", "medium", "large"]
        
        if not os.path.exists(ORIG_SRC_DIR):
            print(f"[!] Error: Original binary directory not found: {ORIG_SRC_DIR}")
            sys.exit(1)

    def parse_summary(self, output):
        """解析 make check 的汇总行"""
        pattern = r"# TOTAL:\s+(\d+).*# PASS:\s+(\d+).*# SKIP:\s+(\d+).*# FAIL:\s+(\d+)"
        match = re.search(pattern, output, re.DOTALL)
        if match:
            return {
                "TOTAL": int(match.group(1)),
                "PASS": int(match.group(2)),
                "FAIL": int(match.group(4))
            }
        return None

    def run_coreutils_test(self, bin_name):
        """为特定二进制运行对应的测试脚本"""
        # 针对校验工具的特殊映射
        actual_test_name = "cksum" if any(x in bin_name for x in ["md5", "sha", "sum", "b2sum"]) else bin_name
        
        try:
            # 修正路径逻辑：从 BUILD_DIR (cwd) 出发，回退一级到 coreutils-9.8 根目录
            # 使用模糊匹配查找 tests/ 下所有包含工具名的 .sh 或 .pl
            # 例如：tests/misc/ls-misc.sh 或 tests/ls/ls-a.sh
            find_cmd = f"find ../tests -name '*{actual_test_name}*.sh' -o -name '*{actual_test_name}*.pl'"
            # print('find_cmd:', find_cmd)
            
            # 执行查找
            files_raw = subprocess.check_output(find_cmd, shell=True, cwd=BUILD_DIR).decode().splitlines()
            
            # print('files_raw:', files_raw)
            
            # 过滤：确保文件名中确实包含该工具（排除 grep 出来的无关项）
            valid_tests = []
            for f in files_raw:
                fname = os.path.basename(f)
                if actual_test_name in fname:
                    valid_tests.append(f)
            # print('valid_tests:', valid_tests)
            
            if not valid_tests:
                # print(f"        [!] No test files found for {bin_name} using pattern *{actual_test_name}*")
                return None
            
            # 执行测试：必须在 BUILD_DIR 下执行 make check
            # TESTS 传入的是相对于 BUILD_DIR 的路径（即以 ../tests 开头的路径）
            check_cmd = f"LC_ALL=C make -k check TESTS='{' '.join(valid_tests)}' SUBDIRS=."
            proc = subprocess.run(check_cmd, shell=True, cwd=BUILD_DIR, capture_output=True, text=True, timeout=60)
            
            return self.parse_summary(proc.stdout)
            
        except Exception as e:
            print(f"        [!] Subprocess Error: {e}")
            return None
    
    def run_coreutils_test_0(self, bin_name):
        """定位并运行特定二进制的 Coreutils 测试集"""
        test_rel_path = f"../tests/{bin_name}"
        check_path = os.path.join(BUILD_DIR, test_rel_path)
        
        # 兼容性处理：md5sum 等工具共用 cksum 测试集
        if not os.path.exists(check_path):
            if any(x in bin_name for x in ["md5", "sha", "sum"]):
                test_rel_path = "../tests/cksum"
            else:
                return None

        try:
            find_cmd = f"find {test_rel_path} -name '*.sh' -o -name '*.pl'"
            files = subprocess.check_output(find_cmd, shell=True, cwd=BUILD_DIR).decode().split()
            if not files: return None
            
            # 执行 LC_ALL=C 确保报错信息一致
            check_cmd = f"LC_ALL=C make -k check TESTS='{' '.join(files)}'"
            proc = subprocess.run(check_cmd, shell=True, cwd=BUILD_DIR, capture_output=True, text=True)
            return self.parse_summary(proc.stdout)
        except:
            return None

    def validate_scale(self, scale_label):
        """遍历特定 scale 文件夹下的文件并对比测试"""
        scale_dir = os.path.join(OBF_ROOT, f"{self.dataset_name}_{scale_label}")
        if not os.path.exists(scale_dir):
            print(f"[W] Skipping scale {scale_label}: Directory {scale_dir} not found.")
            return None

        # 以混淆版目录下的文件为基准
        obf_bins = [f for f in os.listdir(scale_dir) if os.path.isfile(os.path.join(scale_dir, f))]
        
        stats = {"TOTAL": 0, "BASE_P": 0, "BASE_F": 0, "SABRE_P": 0, "SABRE_F": 0}
        
        print(f"[*] Processing scale: {scale_label.upper()} ({len(obf_bins)} files)")

        for b in obf_bins:
            print(f"    [*] Validating {b}...")
            orig_bin_path = os.path.join(ORIG_SRC_DIR, b)
            obf_bin_path = os.path.join(scale_dir, b)
            print(f"        - Original: {orig_bin_path}, Obfuscated: {obf_bin_path}")

            if not os.path.exists(orig_bin_path):
                print(f"    [!] Skipping {b}: Original not found in build dir.")
                continue

            # 1. 运行原版基准 (Baseline)
            # 注意：此时 ORIG_SRC_DIR 下已经是原版，直接跑
            res_base = self.run_coreutils_test(b)
            print(f"        - Baseline result: {res_base if res_base else 'No tests run'}")
            if not res_base: continue

            # 2. 替换为混淆版
            backup_path = orig_bin_path + ".bak"
            shutil.move(orig_bin_path, backup_path)
            shutil.copy(obf_bin_path, orig_bin_path)
            os.chmod(orig_bin_path, 0o755)

            # 3. 运行混淆版测试 (SABRE)
            res_sabre = self.run_coreutils_test(b)

            # 4. 还原原版
            os.remove(orig_bin_path)
            shutil.move(backup_path, orig_bin_path)

            if res_sabre:
                stats["TOTAL"] += res_base["TOTAL"]
                stats["BASE_P"] += res_base["PASS"]
                stats["BASE_F"] += res_base["FAIL"]
                stats["SABRE_P"] += res_sabre["PASS"]
                stats["SABRE_F"] += res_sabre["FAIL"]
                print(f"    - {b}: Base({res_base['PASS']}P/{res_base['FAIL']}F) -> SABRE({res_sabre['PASS']}P/{res_sabre['FAIL']}F)")

        return stats

    def run(self):
        results = []
        for scale in self.scales:
            scale_stats = self.validate_scale(scale)
            if scale_stats:
                is_consistent = "100%" if (scale_stats["BASE_P"] == scale_stats["SABRE_P"] and 
                                           scale_stats["BASE_F"] == scale_stats["SABRE_F"]) else "Diff"
                results.append({
                    "Scale": scale.capitalize(),
                    "Total": scale_stats["TOTAL"],
                    "Base": f"{scale_stats['BASE_P']}P/{scale_stats['BASE_F']}F",
                    "SABRE": f"{scale_stats['SABRE_P']}P/{scale_stats['SABRE_F']}F",
                    "Consistency": is_consistent
                })
        
        self.print_latex(results)

    def print_latex(self, results):
        print("\n" + "="*30 + " LaTeX OUTPUT " + "="*30)
        print("\\begin{table}[htbp]")
        print("  \\centering")
        print("  \\caption{Functional Consistency Across Scales (SABRE vs. Clang-O2)}")
        print("  \\label{tab:consistency_results}")
        print("  \\begin{tabular}{lcccc}")
        print("    \\hline")
        print("    Scale & Total & Base (Clang-O2) & SABRE (Ours) & Consistency \\\\")
        print("    \\hline")
        for r in results:
            print(f"    {r['Scale']} & {r['Total']} & {r['Base']} & {r['SABRE']} & {r['Consistency']} \\\\")
        print("    \\hline")
        print("  \\end{tabular}")
        print("\\end{table}")

if __name__ == "__main__":
    # 使用方式: python test.py rew_bin_coreutils-clang-o2
    # 脚本会自动拼成 ../dataset/rew_bin_coreutils-clang-o2_small 等
    ds_name = sys.argv[1] if len(sys.argv) > 1 else "rew_bin_coreutils-clang-o2"
    validator = SABREScaleValidator(ds_name)
    validator.run()