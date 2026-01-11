# start.py  — маленький лаунчер, який запускає зовнішній execute.py
import os
import sys
import subprocess
from pathlib import Path

def main():
    base_dir = Path(sys.executable).parent.resolve()
    
    script_path = base_dir / "_internal" / "execute.py"
    
    if not script_path.is_file():
        print("═" * 60)
        print("КРИТИЧНА ПОМИЛКА")
        print("Не знайдено файл обробки:")
        print(f"   {script_path}")
        print("Перевірте, чи поклали ви execute.py у папку _internal")
        print("═" * 60)
        sys.exit(1)
    
    # Передаємо всі аргументи командного рядка
    cmd = [sys.executable, str(script_path)] + sys.argv[1:]
    
    try:
        # Запускаємо та чекаємо завершення
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            # Дуже корисно під час розробки — бачимо всі print'и та помилки
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        # Якщо скрипт завершився з помилкою — передаємо той самий код повернення
        sys.exit(e.returncode)
    except Exception as e:
        print("Помилка запуску обробника:")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()