#! /stu01/wumej22/Anaconda3/envs/image/bin/python
# -*- coding: utf-8 -*-

import os
import io
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageColor, ImageDraw, ImageFont

# ==============================================================================
# 1. 字体与 Matplotlib 全局配置
# ==============================================================================

# 尝试加载您的 Noto Sans 字体文件
custom_font_paths = [
    "/home/wumej22/.local/share/fonts/NotoSans-Regular.ttf",
    "/home/wumej22/.local/share/fonts/NotoSans-Bold.ttf",
    "/home/wumej22/.local/share/fonts/NotoSans-Italic.ttf",
    "/home/wumej22/.local/share/fonts/NotoSans-BoldItalic.ttf"
]

for fpath in custom_font_paths:
    if os.path.exists(fpath):
        fm.fontManager.addfont(fpath)

# 设置全局默认参数
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Noto Sans', 'Arial', 'DejaVu Sans']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.default'] = 'regular'

plt.set_loglevel("warning")

# 检查 ReportLab
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4, landscape
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

Image.MAX_IMAGE_PIXELS = 1_000_000_000
try:
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except Exception:
    _RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", Image.BICUBIC)


# ==============================================================================
# 2. 基础辅助函数 (I/O, 颜色, 创建)
# ==============================================================================

def read(path, debug=False):
    if not os.path.exists(path):
        raise FileNotFoundError("找不到文件: {}".format(path))
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if debug:
        print("读取图片: {} | 尺寸: {}".format(path, img.size))
    return img

def save(img, path, debug=False, **kwargs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    
    if 'dpi' in kwargs and ext == '.png':
        dpi_value = kwargs['dpi']
        if isinstance(dpi_value, (int, float)):
            kwargs['dpi'] = (int(dpi_value), int(dpi_value))

    if ext == '.pdf' and REPORTLAB_AVAILABLE:
        save_as_pdf(img, path, debug=debug, **kwargs)
    else:
        img.save(path, **kwargs)
        if debug:
            print("已保存图片到: {}".format(path))

def save_as_pdf(img, path, dpi=300, page_size='A4', margin=None,
                background_color=(255, 255, 255), debug=False, **kwargs):
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab未安装")
    img_pixel_count = img.size[0] * img.size[1]

    if margin is None:
        margin = 0.1 * inch

    # 计算图像在 PDF 中的自然尺寸（单位：点，1点=1/72英寸）
    scale = dpi / 72.0                # 从像素到点的转换因子
    nat_w = img.size[0] / scale        # 图像原始宽度（点）
    nat_h = img.size[1] / scale        # 图像原始高度（点）

    # 确定页面大小
    if isinstance(page_size, str) and page_size.lower() == 'original':
        # 原始尺寸模式：页面大小 = 图像自然大小 + 边距
        pg_w = nat_w + 2 * margin
        pg_h = nat_h + 2 * margin
        pg_sz = (pg_w, pg_h)
        scale_factor = 1.0              # 不缩放
        # 图像绘制位置：左下角 (margin, margin)
        img_x = margin
        img_y = margin
        img_width = nat_w
        img_height = nat_h
    else:
        # 原有逻辑：使用指定的 page_size（字符串或元组）
        page_size_map = {
            'A4': A4,
            'letter': letter,
            'A4_landscape': landscape(A4),
            'letter_landscape': landscape(letter)
        }
        if isinstance(page_size, str):
            pg_sz = page_size_map.get(page_size, A4)
        elif isinstance(page_size, (tuple, list)):
            pg_sz = tuple(page_size)
        else:
            pg_sz = A4

        pg_w, pg_h = pg_sz
        usable_w = pg_w - 2 * margin
        usable_h = pg_h - 2 * margin

        # 计算缩放因子，使图像适应可用区域（保持比例）
        if nat_w > 0 and nat_h > 0:
            scale_factor = min(usable_w / nat_w, usable_h / nat_h)
        else:
            scale_factor = 1

        img_width = nat_w * scale_factor
        img_height = nat_h * scale_factor
        # 居中绘制
        img_x = margin + (usable_w - img_width) / 2
        img_y = margin + (usable_h - img_height) / 2

    # 创建 PDF 画布
    c = canvas.Canvas(path, pagesize=pg_sz)

    # 可选：设置背景色（如果需要）
    if background_color:
        c.setFillColorRGB(*[x/255.0 for x in background_color[:3]])
        c.rect(0, 0, pg_w, pg_h, fill=1, stroke=0)

    # 临时保存图像（reportlab 需要从文件读取）
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        img.save(temp_path, dpi=(dpi, dpi))

        # 在 PDF 上绘制图像
        c.drawImage(temp_path, img_x, img_y,
                    width=img_width, height=img_height,
                    preserveAspectRatio=True, mask='auto')
        c.save()
    except Exception as e:
        raise RuntimeError(f"保存PDF错误: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

def create_blank_image(width, height, color=(255, 255, 255), debug=False):
    """创建空白图片"""
    return Image.new('RGB', (width, height), color)

def _color_to_rgb(color, debug=False):
    if isinstance(color, str):
        try: return tuple(ImageColor.getrgb(color)[:3])
        except: return (0, 0, 0)
    return tuple(color[:3]) if isinstance(color, (list, tuple)) else (0, 0, 0)


# ==============================================================================
# 3. 基础图像处理逻辑 (缩放、裁剪、布局计算) - 【此前遗漏部分已补全】
# ==============================================================================

def resize_image_scale(img: Image.Image, scale: float, resample=_RESAMPLE_LANCZOS) -> Image.Image:
    """仅按比例缩放"""
    if scale is None:
        return img
    s = float(scale)
    if s <= 0:
        raise ValueError(f"scale 必须 > 0，当前: {scale}")
    w, h = img.size
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    return img.resize((nw, nh), resample=resample)

def _clamp_crop_box(left, top, right, bottom, w, h, debug=False):
    l, t = max(0, int(left)), max(0, int(top))
    r, b = min(w, int(right)), min(h, int(bottom))
    if r <= l or b <= t:
        raise ValueError(f"无效的裁剪区域: left={l}, top={t}, right={r}, bottom={b}")
    return l, t, r, b

def crop_image(img, crop_params, mode="pixel", debug=False):
    """
    裁剪单张图片 (Pure Logic)
    """
    try:
        w, h = img.size
        if mode == "ratio":
            # ratio 模式: left/top 是比例, right/bottom 是外侧留白比例
            # 例如 right=0.1 意味着右边切掉 10%
            l = int(crop_params.get('left', 0) * w)
            t = int(crop_params.get('top', 0) * h)
            r_ratio = crop_params.get('right', 0)
            b_ratio = crop_params.get('bottom', 0)
            r = w - int(r_ratio * w)
            b = h - int(b_ratio * h)
        else:
            # pixel 模式: 直接坐标
            l = int(crop_params.get('left', 0))
            t = int(crop_params.get('top', 0))
            r = int(crop_params.get('right', w))
            b = int(crop_params.get('bottom', h))
        
        l, t, r, b = _clamp_crop_box(l, t, r, b, w, h)
        if debug:
            print(f"Crop box: {l}, {t}, {r}, {b}")
        return img.crop((l, t, r, b))
    except Exception as e:
        raise RuntimeError(f"裁剪图片时出错: {e}")

def _calculate_actual_pixel(image_width, image_height, cols_space, rows_space, box_space, space_mode, debug=False):
    """布局计算辅助函数"""
    # 规整输入
    if cols_space is None: cols_space = []
    if rows_space is None: rows_space = []
    if box_space is None: box_space = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}

    # 统一转换为 list 的 list
    def to_2d_list(inp):
        if not inp: return []
        if isinstance(inp, list) and inp and isinstance(inp[0], list): return inp
        return [inp] # 假设是一维，转为 2D 方便处理，或者后续自行展开

    # 像素转换
    def to_px(val, base):
        return int(val * base) if space_mode == 'ratio' else int(val)

    # Box Space
    act_box = {
        'top': to_px(box_space.get('top',0), image_height),
        'bottom': to_px(box_space.get('bottom',0), image_height),
        'left': to_px(box_space.get('left',0), image_width),
        'right': to_px(box_space.get('right',0), image_width),
    }

    # Cols/Rows Space (递归处理)
    # 为了兼容之前的逻辑，这里简单处理：如果是 2D 列表就循环处理，1D 就直接转换
    cols_px = []
    if cols_space and isinstance(cols_space[0], list):
        for row in cols_space:
            cols_px.append([to_px(x, image_width) for x in row])
    elif cols_space:
        cols_px = [to_px(x, image_width) for x in cols_space]

    rows_px = []
    if rows_space and isinstance(rows_space[0], list):
        for row in rows_space:
            rows_px.append([to_px(x, image_height) for x in row])
    elif rows_space:
        rows_px = [to_px(x, image_height) for x in rows_space]

    return cols_px, rows_px, act_box


# ==============================================================================
# 4. 核心渲染引擎：Matplotlib 文字转图片
# ==============================================================================
def render_text_mpl(text, fontsize, color, fontweight="normal", dpi=300):
    """
    完全由 Matplotlib 负责渲染文字，返回透明背景的 PIL Image。
    """
    mpl_color = [c/255.0 for c in color] if isinstance(color, (list, tuple)) else color

    rc_context = {
        "mathtext.fontset": "custom",
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans"],
    }

    if fontweight == "bold":
        rc_context.update({
            "mathtext.rm": "Noto Sans:bold",
            "mathtext.bf": "Noto Sans:bold",
            "mathtext.it": "Noto Sans:bold:italic",
            "font.weight": "bold",
            "mathtext.default": "bf"
        })
    else:
        rc_context.update({
            "mathtext.rm": "Noto Sans",
            "mathtext.bf": "Noto Sans:bold",
            "mathtext.it": "Noto Sans:italic",
            "font.weight": "normal",
            "mathtext.default": "regular"
        })

    with plt.rc_context(rc_context):
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, text, 
                 fontsize=fontsize, 
                 color=mpl_color,
                 fontweight=fontweight,
                 horizontalalignment='center', 
                 verticalalignment='center')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, transparent=True, 
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGBA")


# ==============================================================================
# 5. 图像拼接主逻辑 (merge_images_Row)
# ==============================================================================

def merge_images_Row(
    rows_images,
    cols_space=None, rows_space=None, box_space=None,
    alignment="center",
    background_color=(255, 255, 255),
    space_mode="pixel",
    texts=None,
    draw_ticks=False, tick_step=0.1,
    font_path=None, # 保留参数兼容接口
    debug=False
):
    if not rows_images or not rows_images[0]:
        raise ValueError("rows_images 为空或格式错误")

    nrow = len(rows_images)
    base_w, base_h = rows_images[0][0].size

    # 补全默认参数
    if cols_space is None: cols_space = [[0] * (len(r) - 1) for r in rows_images]
    if rows_space is None: rows_space = [0] * (nrow - 1)
    if box_space is None: box_space = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}

    # 计算像素间距
    cols_px, rows_px, actual_box_space = _calculate_actual_pixel(
        base_w, base_h, cols_space, rows_space, box_space, space_mode
    )

    # 辅助读取间距的闭包
    def get_col_gap(r_idx, c_idx):
        if not cols_px: return 0
        # 检查 cols_px 是 1D 还是 2D
        if isinstance(cols_px[0], list):
            row_gaps = cols_px[r_idx] if r_idx < len(cols_px) else []
            return row_gaps[c_idx] if c_idx < len(row_gaps) else 0
        else:
            # 1D 情况：假设所有行共享
            return cols_px[c_idx] if c_idx < len(cols_px) else 0

    def get_row_gap(r_idx):
        if not rows_px: return 0
        if isinstance(rows_px[0], list): 
            # 异常情况，行间距通常是 1D
            return rows_px[r_idx][0] if r_idx < len(rows_px) and rows_px[r_idx] else 0 
        else:
            return rows_px[r_idx] if r_idx < len(rows_px) else 0

    # 计算宽高
    row_heights = [max(img.height for img in row) for row in rows_images]
    row_widths = []
    for i, row in enumerate(rows_images):
        w = sum(img.width for img in row)
        # 加列间距
        if len(row) > 1:
            gaps_sum = sum(get_col_gap(i, j) for j in range(len(row)-1))
            w += gaps_sum
        row_widths.append(w)
    
    max_w = max(row_widths) if row_widths else 0
    aligns = [alignment]*nrow if isinstance(alignment, str) else alignment

    # Justify 简单处理
    for i in range(nrow):
        if aligns[i] == "justify" and len(rows_images[i]) > 1:
            row_widths[i] = max_w 

    # 画布尺寸
    total_rows_gap = sum(get_row_gap(i) for i in range(nrow-1)) if nrow > 1 else 0
    total_h = sum(row_heights) + total_rows_gap
    fig_w = max_w + actual_box_space['left'] + actual_box_space['right']
    fig_h = total_h + actual_box_space['top'] + actual_box_space['bottom']

    result = Image.new('RGB', (fig_w, fig_h), background_color)

    # 粘贴
    curr_y = actual_box_space['top']
    for i, row in enumerate(rows_images):
        w_row, h_row = row_widths[i], row_heights[i]
        align = aligns[i]
        
        if align == "center":
            cx = actual_box_space['left'] + (fig_w - actual_box_space['left'] - actual_box_space['right'] - w_row) // 2
        elif align == "right":
            cx = fig_w - actual_box_space['right'] - w_row
        else:
            cx = actual_box_space['left']
        
        for j, im in enumerate(row):
            result.paste(im, (cx, curr_y))
            cx += im.width
            if j < len(row) - 1:
                cx += get_col_gap(i, j)
        
        curr_y += h_row
        if i < nrow - 1:
            curr_y += get_row_gap(i)

    # 绘制文本 (Matplotlib Render)
    if texts:
        for key, params in texts.items():
            xc = int(params.get("x", 0.5) * fig_w)
            yc = int(params.get("y", 0.5) * fig_h)
            
            # fontsize 转换
            fs_val = params.get("fontsize", 0.02)
            scale_factor = 72 / 300
            if fs_val < 1: mpl_fontsize = int(fs_val * fig_h * scale_factor)
            else: mpl_fontsize = int(fs_val)

            txt_content = params.get("text", key)
            ha = params.get("ha", "center")
            va = params.get("va", "center")
            color = _color_to_rgb(params.get("color", "black"))
            fontweight = params.get("fontweight", "normal")
            rotation = params.get("rotation", 0)

            try:
                text_img = render_text_mpl(
                    txt_content, fontsize=mpl_fontsize, color=color, 
                    fontweight=fontweight, dpi=300
                )
            except Exception as e:
                print(f"[ERROR] MPL Text Render Failed: {e}")
                continue

            if rotation != 0:
                final_img = text_img.rotate(-rotation, expand=True, resample=Image.BICUBIC)
            else:
                final_img = text_img
            
            rw, rh = final_img.size
            paste_x, paste_y = xc, yc
            
            if ha == "center": paste_x -= rw // 2
            elif ha == "right": paste_x -= rw
            
            if va == "center": paste_y -= rh // 2
            elif va == "bottom": paste_y -= rh
            
            result.paste(final_img, (paste_x, paste_y), final_img)

    # 绘制简单刻度 (PIL)
    if draw_ticks:
        draw = ImageDraw.Draw(result)
        try: font = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
        except: font = ImageFont.load_default()
        steps = int(1 / tick_step) if tick_step > 0 else 0
        for i in range(steps + 1):
            r = i * tick_step
            x = int(r * fig_w)
            draw.line([(x, 0), (x, 5)], fill="black")
            draw.line([(x, fig_h - 5), (x, fig_h)], fill="black")
            draw.text((x, 10), "{:.2f}".format(r), font=font, fill="black")
            if 0 < x < fig_w: draw.line([(x, 0), (x, fig_h)], fill="lightgray")
        for i in range(steps + 1):
            r = i * tick_step
            y = int(r * fig_h)
            draw.line([(0, y), (5, y)], fill="black")
            draw.line([(fig_w - 5, y), (fig_w, y)], fill="black")
            draw.text((10, y), "{:.2f}".format(r), font=font, fill="black")
            if 0 < y < fig_h: draw.line([(0, y), (fig_w, y)], fill="lightgray")

    return result


# ==============================================================================
# 6. 便捷接口 (Wrapper Functions)
# ==============================================================================

def adjust_image_to_ref_canvas(target_img, ref_img, background_color='white', axis=None, align='center'):
    rw, rh = ref_img.size
    tw, th = target_img.size
    nw, nh = max(rw, tw), max(rh, th)
    if axis == 'width': nh = th
    elif axis == 'height': nw = tw
    if (nw, nh) == (tw, th): return target_img.copy()
    
    new_img = Image.new(target_img.mode if target_img.mode!='P' else 'RGB', (nw, nh), background_color)
    align = align.lower()
    xo = 0 if 'left' in align else (nw-tw if 'right' in align else (nw-tw)//2)
    yo = 0 if 'top' in align else (nh-th if 'bottom' in align else (nh-th)//2)
    
    mask = target_img.split()[-1] if 'A' in target_img.getbands() else None
    new_img.paste(target_img, (xo, yo), mask=mask)
    return new_img

def overlay_images(refimg, addimg, x, y, ha="center", va="center", pos_mode="ratio", expand=False, background_color=(255,255,255), scale=None, debug=False):
    if scale: addimg = resize_image_scale(addimg, scale)
    base = refimg.convert("RGBA")
    over = addimg.convert("RGBA")
    bw, bh = base.size
    ow, oh = over.size
    px = int(round(x*bw)) if pos_mode=="ratio" else int(round(x))
    py = int(round(y*bh)) if pos_mode=="ratio" else int(round(y))
    paste_x = px - (ow//2 if ha=="center" else (ow if ha=="right" else 0))
    paste_y = py - (oh//2 if va=="center" else (oh if va=="bottom" else 0))
    
    if not expand:
        base.paste(over, (paste_x, paste_y), mask=over.split()[-1])
        return base.convert(refimg.mode) if refimg.mode!="RGBA" else base
    
    min_x, min_y = min(0, paste_x), min(0, paste_y)
    max_x, max_y = max(bw, paste_x+ow), max(bh, paste_y+oh)
    cw, ch = max_x-min_x, max_y-min_y
    canvas = Image.new("RGBA", (cw, ch), (0,0,0,0))
    if background_color:
        bg = Image.new("RGBA", (cw, ch), background_color if len(background_color)==4 else (*background_color, 255))
        canvas.alpha_composite(bg, (0,0))
    canvas.alpha_composite(base, (-min_x, -min_y))
    canvas.alpha_composite(over, (paste_x-min_x, paste_y-min_y))
    return canvas.convert(refimg.mode) if refimg.mode!="RGBA" else canvas

def crop_image_from_path(input_path, crop_params, mode="pixel", output_path=None, debug=False, **save_kwargs):
    img = read(input_path, debug)
    crp = crop_image(img, crop_params, mode, debug)
    if output_path: save(crp, output_path, debug, **save_kwargs)
    return crp

def merge_images_Row_from_paths(image_rows, input_dir, output_path=None, **kwargs):
    rows_imgs = [[read(os.path.join(input_dir, n) if input_dir else n) for n in r] for r in image_rows]
    res = merge_images_Row(rows_imgs, **kwargs)
    if output_path: save(res, output_path, **kwargs)
    return res



def set_image_width_inch(img, width_inch=21, dpi=300, resample=_RESAMPLE_LANCZOS, debug=False):
    """
    将输入的 PIL Image 等比例缩放到指定物理宽度。
    img : PIL.Image.Image
        输入图片
    width_inch : float
        目标宽度，单位 inch
    dpi : int or float
        输出分辨率
    resample :
        重采样方法
    debug : bool
        是否打印调试信息

    Returns
    -------
    PIL.Image.Image
        调整宽度后的图片
    """

    if img is None:
        raise ValueError("输入 img 不能为空")

    width_inch = float(width_inch)
    dpi = float(dpi)

    if width_inch <= 0:
        raise ValueError(f"width_inch 必须大于 0，当前为: {width_inch}")

    if dpi <= 0:
        raise ValueError(f"dpi 必须大于 0，当前为: {dpi}")

    old_w, old_h = img.size

    target_w = int(round(width_inch * dpi))
    scale = target_w / old_w
    target_h = max(1, int(round(old_h * scale)))

    if debug:
        print(f"原始尺寸: {old_w} x {old_h} px")
        print(f"目标宽度: {width_inch} inch")
        print(f"DPI: {dpi}")
        print(f"目标像素宽度: {target_w} px")
        print(f"缩放后尺寸: {target_w} x {target_h} px")

    return img.resize((target_w, target_h), resample=resample)