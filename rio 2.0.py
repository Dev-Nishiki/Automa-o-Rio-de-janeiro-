from pathlib import Path
from collections import Counter
import os
import unicodedata
import re
import sys
import cv2
import ddddocr
import numpy as np
import pandas as pd
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# ==========================================
# CONFIGURAÇÕES DE DIRETÓRIOS E CONSTANTES
# ==========================================
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

URL_PREFEITURA_RIO = "https://notacarioca.rio.gov.br/capa.aspx"

# XPaths Oficiais de Login e do Captcha
XPATH_CAMPO_USUARIO_LOGIN = "xpath=/html/body/form/div[3]/div[1]/div[6]/div/div/div[2]/div[1]/div[1]/div/div[1]/input[1]"
XPATH_CAMPO_SENHA_LOGIN = "xpath=/html/body/form/div[3]/div[1]/div[6]/div/div/div[2]/div[1]/div[1]/div/div[1]/input[2]"
XPATH_CAMPO_CODIGO_CAPTCHA = "xpath=/html/body/form/div[3]/div[1]/div[6]/div/div/div[2]/div[1]/div[1]/div/div[2]/div/div[2]/div/input"

PLANILHA = BASE_DIR / "Teste RJ.xlsx"
PASTA_RECIBOS = BASE_DIR / "recibos"
PASTA_DISPENSADOS = BASE_DIR / "dispensados"
PASTA_ACESSOS_INVALIDOS = BASE_DIR / "acessos_invalidos"
PASTA_CAPTCHA = BASE_DIR / "outputs"

RECIBO_USAR_SCREENSHOT_EM_VEZ_DE_PDF = True
MAX_TENTATIVAS_LOGIN = 5  
MAX_TENTATIVAS_OCR_CAPTCHA = 15  

_OCR_INSTANCIA = None

def _get_ocr():
    global _OCR_INSTANCIA
    if _OCR_INSTANCIA is None:
        _OCR_INSTANCIA = ddddocr.DdddOcr(show_ad=False)
    return _OCR_INSTANCIA

# ========================================================
# 🛡️ MOTOR DE CAPTCHA COM LIMIAR SECO (PRETO E BRANCO PURODESK)
# ========================================================
def gerar_variantes_captcha(caminho_imagem: Path, salvar_debug: bool = True) -> list[np.ndarray]:
    """Aplica escala de cinza e thresholding para isolar letras pretas em fundo branco."""
    try:
        bytes_da_imagem = caminho_imagem.read_bytes()
        matriz_numpy = np.frombuffer(bytes_da_imagem, np.uint8)
        # 🌟 CONVERTE PARA TONS DE CINZA (Remove variação de cor)
        imagem = cv2.imdecode(matriz_numpy, cv2.IMREAD_GRAYSCALE)
    except Exception:
        imagem = None

    if imagem is None:
        raise FileNotFoundError(f"Nao foi possivel ler a imagem: {caminho_imagem}")

    suavizada = cv2.GaussianBlur(imagem, (3, 3), 0)
    kernel = np.ones((2, 2), np.uint8)
    variantes = [imagem]

    # 1. Variações Dinâmicas por Otsu (Seus filtros originais)
    _, otsu_inv = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variantes.append(cv2.dilate(otsu_inv, kernel, iterations=1))

    adapt_inv = cv2.adaptiveThreshold(
        suavizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )
    variantes.append(cv2.morphologyEx(adapt_inv, cv2.MORPH_OPEN, kernel, iterations=1))

    # 2. 🌟 SUA SUGESTÃO: Limiar Fixo Agressivo (Thresholding manual)
    # Transforma o ruído de fundo em branco puro (255) e letras em preto puro (0)
    _, limiar_puro = cv2.threshold(suavizada, 127, 255, cv2.THRESH_BINARY)
    variantes.append(limiar_puro)

    _, limiar_invertido = cv2.threshold(suavizada, 127, 255, cv2.THRESH_BINARY_INV)
    variantes.append(limiar_invertido)

    if salvar_debug:
        for i, variante in enumerate(variantes):
            saida_debug = caminho_imagem.with_name(f"{caminho_imagem.stem}_preprocessada_{i}.png")
            ok, buf = cv2.imencode(".png", variante)
            if ok:
                saida_debug.write_bytes(buf.tobytes())

    return variantes

def _normalizar_ocr(texto: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", texto.upper())

def extrair_texto_captcha(caminho_imagem: Path, salvar_debug: bool = True) -> str:
    try:
        variantes = gerar_variantes_captcha(caminho_imagem, salvar_debug=salvar_debug)
    except Exception:
        return ""
        
    ocr = _get_ocr()
    candidatos: list[str] = []

    for variante in variantes:
        ok, imagem_png = cv2.imencode(".png", variante)
        if not ok:
            continue
        texto = _normalizar_ocr(ocr.classification(imagem_png.tobytes()))
        
        if len(texto) == 5:
            candidatos.append(texto)

    if not candidatos:
        return ""

    return max(candidatos, key=lambda t: -abs(len(t) - 5))

# ==========================================
# OPERAÇÕES DE ELEMENTOS E NAVEGAÇÃO
# ==========================================
def encontrar_elemento_em_todas_as_frames(page, selectors):
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.is_visible(timeout=1000):
                return locator
        except Exception: pass
        for frame in page.frames:
            try:
                locator = frame.locator(selector).first
                if locator.is_visible(timeout=1000):
                    return locator
            except Exception: pass
    return None

def safe_goto(page, url, max_retries=3):
    for i in range(1, max_retries + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            return True
        except Exception:
            print(f"      [Instabilidade] Erro ao carregar site (Tentativa {i}/{max_retries}). Aguardando...")
            page.wait_for_timeout(5000)
    return False

def recarregar_captcha_link_problemas_imagem(page, usuario_val=None, senha_val=None):
    locators_link = ["text=Clique aqui", "xpath=//a[contains(@href,'location.reload')]", "xpath=//img[contains(@src,'CaptchaImage.aspx')]"]
    link = encontrar_elemento_em_todas_as_frames(page, locators_link)
    if link:
        try:
            link.click(force=True)
            page.wait_for_timeout(1500)  
        except Exception: pass
    if usuario_val and senha_val:
        try:
            page.locator(XPATH_CAMPO_USUARIO_LOGIN).fill(usuario_val)
            page.locator(XPATH_CAMPO_SENHA_LOGIN).fill(senha_val)
        except Exception: pass

def preencher_captcha_automatico(page, usuario_val=None, senha_val=None):
    selectors_codigo = [XPATH_CAMPO_CODIGO_CAPTCHA, "#ctl00_cphCabMenu_ccCodigo_ccCodigo"]
    
    for tentativa in range(1, MAX_TENTATIVAS_OCR_CAPTCHA + 1):
        campo_codigo = encontrar_elemento_em_todas_as_frames(page, selectors_codigo)
        captcha_locator = page.locator('img[src*="CaptchaImage.aspx"]').first

        if campo_codigo and captcha_locator.is_visible(timeout=5000):
            page.wait_for_timeout(1200) 
            
            PASTA_CAPTCHA.mkdir(parents=True, exist_ok=True)
            caminho_captcha = PASTA_CAPTCHA / "captcha_alvo.png"
            
            captcha_locator.screenshot(path=str(caminho_captcha))
            codigo = extrair_texto_captcha(caminho_captcha, salvar_debug=True)

            if codigo and len(codigo) == 5:
                campo_codigo.click()
                campo_codigo.fill(codigo)
                print(f"      [CAPTCHA] Código gerado com limiar puro: {codigo}")
                return codigo

        print(f"      [CAPTCHA] Leitura instável na tentativa {tentativa}. Buscando nova imagem...")
        if tentativa < MAX_TENTATIVAS_OCR_CAPTCHA:
            recarregar_captcha_link_problemas_imagem(page, usuario_val, senha_val)
            
    return ""

def selecionar_competencia(page, valor_competencia):
    selectors = ["#ctl00_cphCabMenu_ddlMes", "select[name='ctl00$cphCabMenu$ddlMes']"]
    campo_mes = encontrar_elemento_em_todas_as_frames(page, selectors)
    if not campo_mes: raise Exception("Dropdown de competência não localizado.")

    if isinstance(valor_competencia, (datetime, pd.Timestamp)):
        mes_num, ano = valor_competencia.month, valor_competencia.year
    else:
        texto = str(valor_competencia).strip()
        match = re.search(r"(\d{1,2})\D+(\d{4})", texto)
        if not match: raise ValueError(f"Formato inválido: {valor_competencia}")
        mes_num, ano = int(match.group(1)), int(match.group(2))

    nomes = {1: "Janeiro", 2: "Fevereiro", 3: "Marco", 4: "Abril", 5: "Maio", 6: "Junho",
             7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"}
    
    texto_alvo = normalizar_texto(f"{nomes.get(mes_num, '')} de {ano}")
    options = campo_mes.locator("option")
    for idx in range(options.count()):
        if texto_alvo in normalizar_texto(options.nth(idx).inner_text()) or options.nth(idx).get_attribute("value") == str(mes_num):
            campo_mes.select_option(value=options.nth(idx).get_attribute("value"))
            return
    raise Exception(f"Competência '{valor_competencia}' não disponível.")

def normalizar_texto(texto):
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c)).lower().strip()

def imprimir_e_salvar_recibo(page, cnpj_referencia):
    cnpj_limpo = re.sub(r"\D", "", str(cnpj_referencia or "")) or "sem_cnpj"
    try:
        page.locator("text=Aguarde... Carregando Guia...").wait_for(state="hidden", timeout=20000)
    except Exception: pass
    try:
        page.locator("text=Comprovante").or_(page.locator("text=Ausência")).wait_for(state="visible", timeout=30000)
    except Exception: pass

    page.wait_for_timeout(2500) 
    destino = PASTA_RECIBOS / f"{cnpj_limpo}.png"
    page.screenshot(path=str(destino), full_page=True)
    return destino

def carregar_dados():
    if not PLANILHA.exists(): raise FileNotFoundError(f"Planilha não encontrada: {PLANILHA}")
    df = pd.read_excel(PLANILHA)
    user_col = next((c for c in df.columns if "usuario" in c.lower() and "senha" not in c.lower()), df.columns[0])
    pass_col = next((c for c in df.columns if "senha" in c.lower()), df.columns[1])
    comp_col = next((c for c in df.columns if "compet" in c.lower()), df.columns[2])
    status_col = next((c for c in df.columns if "status" in c.lower()), "status")
    data_col = next((c for c in df.columns if "data" in c.lower()), "data_entrega")

    for col in [status_col, data_col, "arquivo_recibo"]:
        if col not in df.columns: df[col] = None
        df[col] = df[col].astype(object)
    return df, user_col, pass_col, comp_col, status_col, data_col

# ==========================================
# ORQUESTRAÇÃO PRINCIPAL
# ==========================================
def main():
    df, user_col, pass_col, comp_col, status_col, data_col = carregar_dados()

    for p in [PASTA_RECIBOS, PASTA_DISPENSADOS, PASTA_ACESSOS_INVALIDOS, PASTA_CAPTCHA]:
        p.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False, 
            args=["--start-maximized", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1.0)
        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.accept())

        for i, Player_Row in df.iterrows():
            status_atual = str(Player_Row.get(status_col) or "").lower()
            if any(x in status_atual for x in ["transmitido", "dispensado", "inválido", "invalido"]):
                continue

            usuario_val = str(Player_Row[user_col]).strip()
            senha_val = str(Player_Row[pass_col]).strip()
            competencia_val = Player_Row[comp_col]
            cnpj_linha = str(Player_Row.iloc[0]).strip()
            usuario_limpo = re.sub(r"\D", "", usuario_val) or "sem_nome"

            print(f"\n[Fila] Processando CNPJ: {usuario_val}")
            login_ok = False
            motivo_falha = "Erro de Sistema/Lentidão"

            try:
                for tentativa in range(1, MAX_TENTATIVAS_LOGIN + 1):
                    print(f"   -> Tentativa de Login {tentativa} de {MAX_TENTATIVAS_LOGIN}...")
                    
                    if not safe_goto(page, URL_PREFEITURA_RIO):
                        motivo_falha = "Site fora do ar (Queda de Conexão)"
                        continue
                        
                    try:
                        page.locator("xpath=/html/body/form/div[3]/div[1]/div[3]/div[2]/div/a/img").click(timeout=15000)
                    except Exception:
                        continue

                    page.wait_for_timeout(1000)
                    page.locator(XPATH_CAMPO_USUARIO_LOGIN).fill(usuario_val)
                    page.locator(XPATH_CAMPO_SENHA_LOGIN).fill(senha_val)

                    codigo_captcha = preencher_captcha_automatico(page, usuario_val, senha_val)
                    if not codigo_captcha:
                        motivo_falha = "Falha ao ler Captcha estável"
                        continue

                    btn_entrar = encontrar_elemento_em_todas_as_frames(page, ["#ctl00_cphCabMenu_btEntrar", "input[value='ENTRAR']"])
                    if btn_entrar:
                        btn_entrar.click(force=True)
                        page.wait_for_timeout(2500)

                    html_raw = normalizar_texto(page.content())
                    
                    if "/contribuinte/" in page.url.lower() or "ausenciamovimento.aspx" in html_raw or "localizacaocontribuinte" in page.url.lower():
                        login_ok = True
                        break
                    
                    if "senha web incorreta" in html_raw or "usuario nao cadastrado" in html_raw or "senha incorreta" in html_raw:
                        print("      [Bloqueio] Portal acusou Senha Incorreta. Parando loop.")
                        motivo_falha = "Acesso inválido"
                        break  
                        
                    if "codigo digital nao confere" in html_raw or "captcha invalido" in html_raw:
                        print("      [Erro] Portal rejeitou o código. Indo para o próximo loop...")
                        motivo_falha = "Erro de Captcha"
                        continue

                    if any(msg in html_raw for msg in ["bloqueado", "excesso de tentativas"]):
                        motivo_falha = "Usuário Bloqueado no Portal"
                        break

                if not login_ok:
                    print(f"Resultado Final: {motivo_falha} para {usuario_val}")
                    df.loc[i, status_col] = motivo_falha
                    df.loc[i, data_col] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    page.screenshot(path=str(PASTA_ACESSOS_INVALIDOS / f"{usuario_limpo}.png"))
                    df.to_excel(PLANILHA, index=False)
                    continue

                # TRATAMENTO PARA A TELA DE GEOLOCALIZAÇÃO/MAPA DO PRESTADOR
                try:
                    html_raw = normalizar_texto(page.content())
                    if "localizacaocontribuinte.aspx" in page.url.lower() or "localizacao do prestador" in html_raw:
                        btn_fechar_mapa = encontrar_elemento_em_todas_as_frames(page, ["text=Fechar", "xpath=//a[contains(text(),'Fechar')]"])
                        if btn_fechar_mapa:
                            btn_fechar_mapa.click(force=True)
                            page.wait_for_timeout(2500)  
                except Exception: pass

                # Processamento pós-login tradicional
                html_raw = normalizar_texto(page.content())
                if "ausencia de movimento" not in html_raw:
                    df.loc[i, status_col] = "Dispensado"
                    page.screenshot(path=str(PASTA_DISPENSADOS / f"{usuario_limpo}.png"))
                    df.to_excel(PLANILHA, index=False)
                    continue

                try:
                    aba_ausencia = encontrar_elemento_em_todas_as_frames(page, ["a[href='/contribuinte/ausenciamovimento.aspx']"])
                    if aba_ausencia: aba_ausencia.click(force=True)

                    selecionar_competencia(page, competencia_val)

                    btn_confirmar = encontrar_elemento_em_todas_as_frames(page, ["#ctl00_cphCabMenu_btConfirmar", "input[value='CONFIRMAR']"])
                    if btn_confirmar: btn_confirmar.click(force=True)

                    btn_imprimir = encontrar_elemento_em_todas_as_frames(page, ["#ctl00_cphCabMenu_gvAusenciaMovimento_ctl02_imbImprimir", "input[title*='Imprimir']"])
                    if btn_imprimir: btn_imprimir.click(force=True)

                    comprovante = imprimir_e_salvar_recibo(page, cnpj_linha)

                    df.loc[i, status_col] = "Transmitido"
                    if comprovante: df.loc[i, "arquivo_recibo"] = comprovante.name
                    print(f"Sucesso: Empresa {usuario_val} processada!")
                except Exception as inner_error:
                    df.loc[i, status_col] = f"Erro: {str(inner_error)[:35]}"

                df.loc[i, data_col] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                df.to_excel(PLANILHA, index=False)
                
            except Exception as total_err:
                print(f"   [Erro Crítico de Lentidão] Pulando devido a instabilidade externa: {total_err}")
                df.loc[i, status_col] = "Erro de Instabilidade do Site"
                df.to_excel(PLANILHA, index=False)

        browser.close()

if __name__ == "__main__":
    main()