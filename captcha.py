from pathlib import Path
import re

import cv2
import ddddocr
import numpy as np
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


URL = "https://notacarioca.rio.gov.br/capa.aspx"
OUTPUT_DIR = Path("outputs")
CAPTCHA_FILE = OUTPUT_DIR / "captcha.png"
PAGE_FILE = OUTPUT_DIR / "pagina_login.png"


def gerar_variantes_captcha(caminho_imagem: Path, salvar_debug: bool = False) -> list[np.ndarray]:
    """Gera variacoes da imagem para melhorar a taxa de OCR."""
    imagem = cv2.imread(str(caminho_imagem), cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        raise FileNotFoundError(f"Nao foi possivel abrir a imagem: {caminho_imagem}")

    suavizada = cv2.GaussianBlur(imagem, (3, 3), 0)
    kernel = np.ones((2, 2), np.uint8)
    variantes = [imagem]

    _, otsu_inv = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variantes.append(cv2.dilate(otsu_inv, kernel, iterations=1))

    adapt_inv = cv2.adaptiveThreshold(
        suavizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )
    variantes.append(cv2.morphologyEx(adapt_inv, cv2.MORPH_OPEN, kernel, iterations=1))

    _, otsu = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(otsu)

    if salvar_debug:
        for i, variante in enumerate(variantes):
            saida_debug = caminho_imagem.with_name(
                f"{caminho_imagem.stem}preprocessada{i}.png"
            )
            cv2.imwrite(str(saida_debug), variante)

    return variantes


def _normalizar_texto(texto: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", texto.upper())


def extrair_texto_captcha(caminho_imagem: Path, salvar_debug: bool = False) -> str:
    """Extrai texto do captcha com OCR local (ddddocr)."""
    variantes = gerar_variantes_captcha(caminho_imagem, salvar_debug=salvar_debug)
    ocr = ddddocr.DdddOcr(show_ad=False)
    candidatos: list[str] = []

    for variante in variantes:
        ok, imagem_png = cv2.imencode(".png", variante)
        if not ok:
            continue
        texto = _normalizar_texto(ocr.classification(imagem_png.tobytes()))
        if texto:
            candidatos.append(texto)

    if not candidatos:
        return ""

    # O campo de codigo aceita no maximo 5 caracteres (maxlength=5).
    return max(candidatos, key=lambda t: -abs(len(t) - 5))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)

        # O botão exibido na página inicial ("Entrar no Sistema").
        page.locator('img[title="Entrar no Sistema"]').first.click()
        page.wait_for_load_state("domcontentloaded")

        # Salva screenshot da página de login para conferência.
        page.screenshot(path=str(PAGE_FILE), full_page=True)

        # Captura somente a imagem do CAPTCHA.
        captcha_locator = page.locator('img[src*="CaptchaImage.aspx"]').first
        captcha_locator.wait_for(state="visible", timeout=15000)
        captcha_locator.screenshot(path=str(CAPTCHA_FILE))

        print(f"Screenshot da página salvo em: {PAGE_FILE.resolve()}")
        print(f"Screenshot do CAPTCHA salvo em: {CAPTCHA_FILE.resolve()}")
        texto_captcha = extrair_texto_captcha(CAPTCHA_FILE, salvar_debug=True)
        if texto_captcha:
            print(f"Texto detectado no CAPTCHA: {texto_captcha}")
            codigo = texto_captcha[:5]
            if len(texto_captcha) > 5:
                print(f"Aviso: OCR retornou {len(texto_captcha)} caracteres; usando os 5 primeiros: {codigo}")
            campo_codigo = page.locator("#ctl00_cphCabMenu_ccCodigo_ccCodigo")
            campo_codigo.wait_for(state="visible", timeout=15000)
            campo_codigo.click()
            campo_codigo.fill(codigo)
            print(f"Código preenchido no campo de login: {codigo}")
        else:
            print("Não foi possível detectar texto no CAPTCHA.")

        try:
            # Mantém a janela aberta para preenchimento manual, se você quiser.
            page.wait_for_timeout(20000)
        except PlaywrightTimeoutError:
            pass
        finally:
            browser.close()

if __name__ == "__main__":
    main()