import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import sympy as sp
from sympy import symbols, sympify, sin, cos, tan, exp, log, diff, solve, limit, oo, simplify, Poly
import re

# 값 포맷터: 소수 첫째 자리, 유리수는 p/q로, pi는 'π'로 표현
def format_val(v):
    try:
        # 이미 sympy 객체면 사용
        if isinstance(v, sp.Expr):
            ns = sp.nsimplify(v, [sp.pi])
            if ns.is_Rational:
                p = ns.p
                q = ns.q
                if q == 1:
                    return f"{float(p):.1f}"
                return f"{p}/{q}"
            if ns.has(sp.pi):
                s = str(ns)
                s = s.replace('pi', 'π')
                return s
            try:
                return f"{float(ns.evalf()):.1f}"
            except Exception:
                return str(ns)

        # numpy numbers or Python floats/ints
        if isinstance(v, (float, int, np.floating, np.integer)):
            # 시도: pi 또는 유리수로 근사
            try:
                ns = sp.nsimplify(v, [sp.pi])
                if ns.is_Rational:
                    p = ns.p
                    q = ns.q
                    if q == 1:
                        return f"{float(p):.1f}"
                    return f"{p}/{q}"
                if ns.has(sp.pi):
                    s = str(ns)
                    s = s.replace('pi', 'π')
                    return s
            except Exception:
                pass
            return f"{float(v):.1f}"

        # 문자열 또는 기타: sympify 시도
        try:
            expr = sp.sympify(v)
            return format_val(expr)
        except Exception:
            return str(v)
    except Exception:
        try:
            return f"{float(v):.1f}"
        except Exception:
            return str(v)

st.set_page_config(page_title="함수 분석 도구", layout="wide")
st.title("🔍 함수 분석 도구")

# 한글 폰트 등록: fonts 폴더의 ttf 파일을 등록하고 NanumGothic 사용
font_dir = os.path.join(os.getcwd(), 'fonts')
if os.path.isdir(font_dir):
    for fname in os.listdir(font_dir):
        if fname.lower().endswith('.ttf'):
            fm.fontManager.addfont(os.path.join(font_dir, fname))
    plt.rcParams['font.family'] = 'NanumGothic'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

st.markdown("""
함수를 입력하면 다음 정보를 분석합니다:
- 함수의 원형 (기본 함수 형태)
- 변환 (평행이동, 대칭이동)
- 점근선
- x절편, y절편
- 함수 그래프
""")

st.markdown("---")

# 함수 입력
col1, col2 = st.columns([2, 1])

if 'func_input' not in st.session_state:
    st.session_state.func_input = "x**2 - 2*x + 1"

with col1:
    func_input = st.text_input(
        "함수를 입력하세요",
        key="func_input",
        help="예: x**2 - 2*x + 1, sin(x), exp(x), log(x), 1/(x-2)"
    )

with col2:
    if st.button("분석", use_container_width=True):
        st.session_state.analyze = True

# 함수 분석 실행
if 'analyze' not in st.session_state:
    st.session_state.analyze = False

if st.session_state.analyze:
    try:
        x = symbols('x', real=True)
        
        # 표현식, 포맷, 그래프 보조 함수
        def format_expr(expr):
            try:
                expr = sp.simplify(expr)
                s = str(expr)
                return s.replace('pi', 'π')
            except Exception:
                return str(expr)

        def safe_plot_expr(ax, expr, x_vals, color='blue', label=None):
            y = sp.lambdify(x, expr, 'numpy')(x_vals)
            y = np.array(y, dtype=np.complex128)
            y_plot = np.where(np.isfinite(y.real) & (np.abs(y.imag) < 1e-8) & (np.abs(y.real) < 1e6), y.real, np.nan)
            ax.plot(x_vals, y_plot, color=color, linewidth=2, label=label)
            ax.axhline(y=0, color='k', linewidth=0.8)
            ax.axvline(x=0, color='k', linewidth=0.8)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)

        def detect_trig_shift(expr):
            notes = []
            if not isinstance(expr, sp.Function):
                return notes
            arg = expr.args[0]
            delta = sp.simplify(arg - x)
            if delta.is_Number and delta != 0:
                shift_amount = -delta
                direction = '오른쪽' if shift_amount > 0 else '왼쪽'
                notes.append(f"x축으로 {direction} {format_val(abs(shift_amount))}만큼 평행이동")
            vertical = sp.simplify(expr - expr.func(arg))
            if vertical.is_Number and vertical != 0:
                direction = '위' if vertical > 0 else '아래'
                notes.append(f"y축으로 {direction} {format_val(abs(vertical))}만큼 평행이동")
            if not notes:
                notes.append('평행이동 없음')
            return notes

        def detect_symmetry(expr):
            try:
                if sp.simplify(expr - expr.subs(x, -x)) == 0:
                    return '짝함수 (y축 대칭)'
                if sp.simplify(expr + expr.subs(x, -x)) == 0:
                    return '기함수 (원점 대칭)'
            except Exception:
                pass
            return '특별한 대칭 없음'

        # 함수를 sympy 식으로 변환 (심볼 x 명시적으로 전달)
        func_expr = sympify(func_input, locals={'x': x})
        
        st.success("✅ 함수가 올바르게 입력되었습니다!")
        
        # ========== 분석 데이터 계산 (모든 탭에서 사용) ==========
        # 정의역 구간 설정 (사이드바에서 미리 설정)
        x_min_default = -5.0
        x_max_default = 5.0
        
        # 극값, 변곡점, 점근선 계산 (탭 독립적으로 수행)
        extrema = []
        inflections = []
        horizontal_asymptotes = []
        vertical_asymptotes = []
        f_prime = None
        f_double_prime = None
        
        # 도함수 계산
        try:
            f_prime = diff(func_expr, x)
            f_double_prime = diff(f_prime, x)
        except Exception:
            pass
        
        # 극값 계산
        if f_prime is not None:
            try:
                critical_points = solve(f_prime, x)
                for point in critical_points:
                    try:
                        # 숫자로 변환
                        if hasattr(point, 'is_real') and point.is_real:
                            point_val = float(point)
                        else:
                            point_val = float(point.evalf())
                        
                        # 함수값 계산
                        y_point = func_expr.subs(x, point)
                        if hasattr(y_point, 'evalf'):
                            y_val = float(y_point.evalf())
                        else:
                            y_val = float(y_point)
                        
                        # 2차 도함수 값 계산
                        if f_double_prime is not None:
                            second_val = f_double_prime.subs(x, point)
                            if hasattr(second_val, 'evalf'):
                                second_val_float = float(second_val.evalf())
                            else:
                                second_val_float = float(second_val)
                            
                            # 극값 판정
                            if second_val_float > 0.0001:
                                extrema.append((point_val, y_val, 'min'))
                            elif second_val_float < -0.0001:
                                extrema.append((point_val, y_val, 'max'))
                        else:
                            # 2차 도함수 없으면 임계점만 저장
                            extrema.append((point_val, y_val, 'unknown'))
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass
        
        # 변곡점 계산
        if f_double_prime is not None:
            try:
                inflection_points = solve(f_double_prime, x)
                for point in inflection_points:
                    try:
                        # 숫자로 변환
                        if hasattr(point, 'is_real') and point.is_real:
                            point_val = float(point)
                        else:
                            point_val = float(point.evalf())
                        
                        # 함수값 계산
                        y_point = func_expr.subs(x, point)
                        if hasattr(y_point, 'evalf'):
                            y_val = float(y_point.evalf())
                        else:
                            y_val = float(y_point)
                        inflections.append((point_val, y_val))
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass
        
        # 수평 점근선 계산
        try:
            limit_pos_inf = limit(func_expr, x, oo)
            limit_neg_inf = limit(func_expr, x, -oo)
            if limit_pos_inf.is_finite:
                try:
                    horizontal_asymptotes.append(float(limit_pos_inf.evalf()))
                except:
                    pass
            if limit_neg_inf.is_finite:
                try:
                    val = float(limit_neg_inf.evalf())
                    if val not in horizontal_asymptotes:
                        horizontal_asymptotes.append(val)
                except:
                    pass
        except Exception:
            pass
        
        # 수직 점근선 계산
        try:
            if '/' in func_input:
                numer, denom = sp.fraction(func_expr)
                denom_zeros = solve(denom, x)
                for zero in denom_zeros:
                    try:
                        if hasattr(zero, 'is_real') and zero.is_real:
                            z_val = float(zero)
                        else:
                            z_val = float(zero.evalf())
                        vertical_asymptotes.append(z_val)
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

        def is_simple_exp(expr):
            if expr.func is exp:
                return True
            if expr.is_Mul:
                exp_parts = [arg for arg in expr.args if arg.func is exp]
                return len(exp_parts) == 1 and all(arg.is_Number or arg.func is exp for arg in expr.args)
            return False

        def is_simple_log(expr):
            if expr.func is log:
                return True
            if expr.is_Mul:
                log_parts = [arg for arg in expr.args if arg.func is log]
                return len(log_parts) == 1 and all(arg.is_Number or arg.func is log for arg in expr.args)
            return False

        def is_transcendental_expr(expr):
            if expr.has(exp) or expr.has(log) or expr.has(sin) or expr.has(cos) or expr.has(tan):
                if is_simple_exp(expr) or is_simple_log(expr):
                    return False
                return True
            return False

        # 탭 생성 (특수점 탭 제거하고 함수 분석에 통합)
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 그래프", "🔍 함수 분석 결과", "📈 변환 분석", "📋 상세 정보"]
        )
        
        # ============= 그래프 탭 =============
        with tab1:
            st.subheader("함수의 그래프")
            
            # 정의역 구간 설정
            col1, col2, col3 = st.columns(3)
            with col1:
                x_min = st.number_input("정의역 구간 설정 - 최소값", value=x_min_default)
            with col2:
                x_max = st.number_input("정의역 구간 설정 - 최대값", value=x_max_default)
            with col3:
                y_auto = st.checkbox("y축 자동 조정", value=True)
            
            try:
                # 함수를 그릴 수 있도록 변환
                func_lambda = sp.lambdify(x, func_expr, 'numpy')
                
                # x 데이터 생성
                x_vals = np.linspace(x_min, x_max, 2000)
                y_vals = func_lambda(x_vals)
                y_vals = np.array(y_vals, dtype=np.complex128)

                # 실수부 기준으로 유한한 값 탐지 (극단값은 비유한 것으로 간주)
                y_real = y_vals.real
                finite_mask = np.isfinite(y_real) & (np.abs(y_real) < 1e6) & (np.abs(y_vals.imag) < 1e-8)

                # 연속된 비유(무한/NaN) 구간을 찾아 각 구간의 중앙을 수직 점근선으로 추가
                false_idx = np.where(~finite_mask)[0]
                if false_idx.size > 0:
                    # 그룹화: 연속된 인덱스들을 묶음
                    groups = np.split(false_idx, np.where(np.diff(false_idx) != 1)[0] + 1)
                    for g in groups:
                        if g.size == 0:
                            continue
                        # 구간 중앙 x 좌표
                        x_asym_center = float(x_vals[g].mean())
                        # 중복 방지 (기존 점근선과 가까우면 무시)
                        if all(abs(x_asym_center - xa) > 1e-6 for xa in vertical_asymptotes):
                            vertical_asymptotes.append(x_asym_center)

                # 플롯용: 비유한 지점은 NaN으로 채워 그래프가 끊기게 함
                y_plot_full = np.where(finite_mask, y_real, np.nan)
                # NaN이 아닌 실제 플롯 값들 (y축 자동/수동 처리 시 사용)
                finite_y_values = y_plot_full[~np.isnan(y_plot_full)]

                # 큰 급변(아직 유한값이지만 점근선 근처에서 발생)을 검출하여 플롯을 끊음
                if finite_y_values.size > 1:
                    med_diff = float(np.nanmedian(np.abs(np.diff(finite_y_values))))
                else:
                    med_diff = 0.0
                # 기준 임계값: 작은 변동은 무시하고 큰 스파이크는 분리
                jump_thresh = max(50.0, med_diff * 40.0)

                discontinuity = np.zeros_like(y_plot_full, dtype=bool)
                for i in range(len(y_plot_full) - 1):
                    a = y_plot_full[i]
                    b = y_plot_full[i + 1]
                    if not np.isnan(a) and not np.isnan(b):
                        if abs(a - b) > jump_thresh:
                            discontinuity[i] = True
                            discontinuity[i + 1] = True

                # 큰 점프 구간을 NaN으로 만들어 선 연결이 되지 않도록 함
                if np.any(discontinuity):
                    y_plot_full[discontinuity] = np.nan

                    # 연속된 discontinuity 인덱스 그룹을 찾아 각 구간 중심에 점근선 추가
                    disc_idx = np.where(discontinuity)[0]
                    groups = np.split(disc_idx, np.where(np.diff(disc_idx) != 1)[0] + 1)
                    for g in groups:
                        if g.size == 0:
                            continue
                        x_asym_center = float(x_vals[g].mean())
                        if all(abs(x_asym_center - xa) > 1e-6 for xa in vertical_asymptotes):
                            vertical_asymptotes.append(x_asym_center)

                if np.any(finite_mask):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(x_vals, y_plot_full, 'b-', linewidth=2, label=f'f(x) = {func_input}')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='k', linewidth=0.8)
                    ax.axvline(x=0, color='k', linewidth=0.8)
                    ax.set_xlabel('x')
                    ax.set_ylabel('f(x)', rotation=90)
                    ax.set_title(f'함수의 그래프: f(x) = {func_input}')
                    
                    # 점근선 표시 (점선)
                    for y_asym in horizontal_asymptotes:
                        ax.axhline(y=y_asym, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, label='수평 점근선' if y_asym == horizontal_asymptotes[0] else '')

                    for x_asym in vertical_asymptotes:
                        if x_min <= x_asym <= x_max:
                            ax.axvline(x=x_asym, color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label='수직 점근선' if x_asym == vertical_asymptotes[0] else '')
                    
                    # 극값 표시
                    if extrema:
                        xs = [p[0] for p in extrema if x_min <= p[0] <= x_max]
                        ys = [p[1] for p in extrema if x_min <= p[0] <= x_max]
                        if xs and ys:
                            ax.scatter(xs, ys, color='red', s=120, zorder=5, marker='o', 
                                      edgecolors='darkred', linewidth=2, label='극값')
                    
                    # 변곡점 표시
                    if inflections:
                        xs = [p[0] for p in inflections if x_min <= p[0] <= x_max]
                        ys = [p[1] for p in inflections if x_min <= p[0] <= x_max]
                        if xs and ys:
                            ax.scatter(xs, ys, color='purple', s=120, zorder=5, marker='^', 
                                      edgecolors='indigo', linewidth=2, label='변곡점')
                    
                    # 범례 정리
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
                    
                    # y축 범위 조정
                    # 플롯에서 큰 점프 부분을 NaN으로 처리했으므로, 여기서 유한값을 재계산
                    finite_y_values = y_plot_full[~np.isnan(y_plot_full)]
                    if y_auto:
                        # 극값과 변곡점을 포함한 y값 범위 계산
                        all_y_vals = list(finite_y_values)
                        
                        # 범위 내 극값 추가
                        all_y_vals.extend([p[1] for p in extrema if x_min <= p[0] <= x_max])
                        
                        # 범위 내 변곡점 추가
                        all_y_vals.extend([p[1] for p in inflections if x_min <= p[0] <= x_max])
                        
                        # y축 범위 계산: 극단값 영향을 줄이기 위해 2~98 퍼센타일 사용
                        if len(all_y_vals) > 0:
                            try:
                                low, high = np.nanpercentile(all_y_vals, [2, 98])
                                if not np.isfinite(low) or not np.isfinite(high):
                                    raise ValueError
                                y_min_data = float(low)
                                y_max_data = float(high)
                            except Exception:
                                y_min_data = float(np.min(all_y_vals))
                                y_max_data = float(np.max(all_y_vals))

                            y_range = y_max_data - y_min_data
                            if y_range < 0.1:
                                y_range = 1.0
                            margin = y_range * 0.15
                            y_min_plot = y_min_data - margin
                            y_max_plot = y_max_data + margin
                            ax.set_ylim(y_min_plot, y_max_plot)
                    else:
                        if finite_y_values.size > 0:
                            default_y_min = float(np.min(finite_y_values))
                            default_y_max = float(np.max(finite_y_values))
                        else:
                            default_y_min = -10.0
                            default_y_max = 10.0
                        y_min = st.number_input("y축 최소값", value=default_y_min)
                        y_max = st.number_input("y축 최대값", value=default_y_max)
                        # 입력값 검증: 같거나 최소값이 최대값보다 크면 자동 보정
                        try:
                            y_min_f = float(y_min)
                            y_max_f = float(y_max)
                            if y_min_f >= y_max_f:
                                # swap if reversed
                                y_min_f, y_max_f = min(y_min_f, y_max_f), max(y_min_f, y_max_f)
                                if y_min_f == y_max_f:
                                    y_min_f -= 1.0
                                    y_max_f += 1.0
                            ax.set_ylim(y_min_f, y_max_f)
                        except Exception:
                            # fallback to safe defaults
                            ax.set_ylim(-10.0, 10.0)
                    
                    st.pyplot(fig)
                else:
                    st.warning("주어진 범위에서 그려진 그래프가 없습니다.")
            except Exception as e:
                st.error(f"그래프 표시 중 오류: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        # ============= 함수 분석 탭 =============
        with tab2:
            st.subheader("함수 분석 결과")
            
            # 1. 원형 함수 및 기본 정보
            st.write("### 📋 함수 정보")
            
            func_str = str(func_expr)
            func_type = "기타 함수"
            
            info_data = []
            
            # 함수 유형 판정
            if func_expr.is_polynomial():
                poly = sp.Poly(func_expr, x)
                degree = poly.degree()
                func_type = f"{degree}차 다항 함수"
                info_data.append(["함수 유형", func_type])
                info_data.append(["최고차항", f"{poly.LC()}·x^{degree}"])
                
            elif 'sin' in func_str or 'cos' in func_str or 'tan' in func_str:
                func_type = "삼각 함수"
                info_data.append(["함수 유형", func_type])
                if 'sin' in func_str:
                    info_data.append(["기본 함수", "sin(x)"])
                elif 'cos' in func_str:
                    info_data.append(["기본 함수", "cos(x)"])
                else:
                    info_data.append(["기본 함수", "tan(x)"])
                    
            elif is_transcendental_expr(func_expr):
                func_type = "초월 함수"
                info_data.append(["함수 유형", func_type])
                info_data.append(["기본 함수", "초월 함수"])

            elif is_simple_exp(func_expr):
                func_type = "지수 함수"
                info_data.append(["함수 유형", func_type])
                info_data.append(["기본 함수", "e^x"])
                
            elif is_simple_log(func_expr):
                func_type = "로그 함수"
                info_data.append(["함수 유형", func_type])
                info_data.append(["기본 함수", "ln(x) 또는 log(x)"])
                
            else:
                info_data.append(["함수 유형", func_type])
            
            # 대칭성
            try:
                func_f_neg = func_expr.subs(x, -x)
                
                if simplify(func_expr - func_f_neg) == 0:
                    info_data.append(["대칭성", "우함수 (짝함수) - y축 대칭"])
                elif simplify(func_expr + func_f_neg) == 0:
                    info_data.append(["대칭성", "기함수 (홀함수) - 원점 대칭"])
                else:
                    info_data.append(["대칭성", "특별한 대칭성 없음"])
            except:
                pass
            
            st.table(info_data)
            
            # 2. 도함수
            st.write("### 🔢 미분")
            derivative_data = []
            
            if f_prime is not None:
                derivative_data.append(["1차 도함수", str(f_prime)])
            else:
                derivative_data.append(["1차 도함수", "계산 불가"])
            
            if f_double_prime is not None:
                derivative_data.append(["2차 도함수", str(f_double_prime)])
            else:
                derivative_data.append(["2차 도함수", "계산 불가"])
            
            st.table(derivative_data)
            
            # 3. 특수점 정보 (표 형식)
            st.write("### 📍 특수점")
            
            # x절편, y절편
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**x절편 (근)**")
                try:
                    zeros = solve(func_expr, x)
                    if zeros:
                        zeros_list = []
                        for zero in zeros:
                            try:
                                zero_val = complex(zero)
                                if abs(zero_val.imag) < 1e-10:  # 실근만
                                    zeros_list.append(format_val(zero_val.real))
                            except:
                                try:
                                    zeros_list.append(format_val(zero.evalf()))
                                except:
                                    zeros_list.append(format_val(zero))
                        if zeros_list:
                            st.write(", ".join(zeros_list))
                        else:
                            st.write("실수 근 없음")
                    else:
                        st.write("실수 근 없음")
                except Exception:
                    st.write("계산 불가")
            
            with col2:
                st.write("**y절편**")
                try:
                    y_intercept = func_expr.subs(x, 0)
                    st.write(format_val(y_intercept))
                except:
                    st.write("계산 불가")
            
            # 극값 표
            st.write("**극값**")
            if extrema:
                extrema_table = []
                for ex in extrema:
                    if ex[2] == 'max':
                        extrema_table.append(["극대점", format_val(ex[0]), format_val(ex[1])])
                    elif ex[2] == 'min':
                        extrema_table.append(["극소점", format_val(ex[0]), format_val(ex[1])])
                
                if extrema_table:
                    st.table(extrema_table)
                else:
                    st.write("극값 없음")
            else:
                st.write("극값 없음")
            
            # 변곡점 표
            st.write("**변곡점**")
            if inflections:
                inflection_table = []
                for inf in inflections:
                    inflection_table.append([format_val(inf[0]), format_val(inf[1])])
                
                st.table(inflection_table)
            else:
                st.write("변곡점 없음")
            
            # 점근선
            st.write("**점근선**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("*수평 점근선*")
                if horizontal_asymptotes:
                    asymptote_list = [f"y = {format_val(h)}" for h in horizontal_asymptotes]
                    st.write("\n".join(asymptote_list))
                else:
                    st.write("없음")
            
            with col2:
                st.write("*수직 점근선*")
                if vertical_asymptotes:
                    asymptote_list = [f"x = {format_val(v)}" for v in vertical_asymptotes]
                    st.write("\n".join(asymptote_list))
                else:
                    st.write("없음")
        
        # ============= 변환 분석 탭 =============
        with tab3:
            st.subheader("함수의 변환 분석")
            st.write("### 원형에서의 변환")
            
            func_str = str(func_expr)
            base_func = None
            base_expr = None
            transform_notes = []
            symmetry_note = detect_symmetry(func_expr)

            # 1. 다항 함수 분석
            if func_expr.is_polynomial():
                poly = Poly(func_expr, x)
                degree = poly.degree()
                base_func = f"y = x^{degree}"
                base_expr = x**degree if degree >= 1 else None
                st.write(f"**원형 함수**: {base_func}")
                
                if degree == 1:
                    # 1차 함수: y = mx + b에서 y = x로의 변환
                    coeffs = poly.all_coeffs()
                    if len(coeffs) == 2:
                        m, b = coeffs
                        st.write(f"**함수**: f(x) = {format_val(m)}x + {format_val(b)}")
                        st.write(f"- 기울기: {format_val(m)}")
                        if b > 0:
                            st.write(f"- y축으로 {format_val(b)}만큼 위로 평행이동")
                        elif b < 0:
                            st.write(f"- y축으로 {format_val(abs(b))}만큼 아래로 평행이동")
                
                elif degree == 2:
                    # 2차 함수: 완전제곱식
                    coeffs = poly.all_coeffs()
                    if len(coeffs) == 3:
                        a_coef, b_coef, c_coef = coeffs
                        h = -b_coef / (2*a_coef)
                        k = func_expr.subs(x, h)
                        
                        st.write(f"**표준형**: f(x) = {format_val(a_coef)}(x - ({format_val(h)}))² + ({format_val(k)})")
                        st.write(f"- **평행이동**:")
                        if h > 0:
                            st.write(f"  - x축으로 {format_val(h)}만큼 오른쪽")
                        elif h < 0:
                            st.write(f"  - x축으로 {format_val(abs(h))}만큼 왼쪽")
                        if k > 0:
                            st.write(f"  - y축으로 {format_val(k)}만큼 위")
                        elif k < 0:
                            st.write(f"  - y축으로 {format_val(abs(k))}만큼 아래")
                        
                        if a_coef < 0:
                            st.write(f"- **대칭이동**: x축에 대해 대칭반사")
                        # 2차 함수의 축대칭 특성 추가 설명
                        st.write("- **대칭성**: 2차 함수는 축대칭(직선에 대한 대칭)을 가집니다. 일반적으로 대칭축은 x = h 입니다.")
                
                elif degree == 3:
                    # 3차 함수
                    st.write(f"**함수**: f(x) = {format_expr(func_expr)}")
                    # 간단한 형태로 표시
                    coeffs = poly.all_coeffs()
                    if len(coeffs) >= 2:
                        st.write(f"- 최고차 계수: {coeffs[0]}")
                        if coeffs[0] > 0:
                            st.write(f"- x → ∞일 때 f(x) → ∞, x → -∞일 때 f(x) → -∞")
                        else:
                            st.write(f"- x → ∞일 때 f(x) → -∞, x → -∞일 때 f(x) → ∞")
                    st.write("- 3차 함수는 변곡점을 기준으로 점대칭을 가집니다.")
                
                else:
                    # n차 함수
                    st.write(f"**함수**: f(x) = {format_expr(func_expr)}")
                    coeffs = poly.all_coeffs()
                    st.write(f"- 차수: {degree}")
                    st.write(f"- 최고차 계수: {coeffs[0]}")
            
            # 2. 삼각 함수 분석
            elif 'sin' in func_str:
                base_func = "y = sin(x)"
                base_expr = sp.sin(x)
                st.write(f"**원형 함수**: {base_func}")
                st.write(f"**함수**: f(x) = {format_expr(func_expr)}")
                
                # 진폭, 주기, 평행이동 분석
                st.write("- **특징**:")
                st.write("  - 정의역: 모든 실수")
                st.write("  - 치역: [-1, 1]")
                st.write("  - 주기: 2π")
            
            elif 'cos' in func_str:
                base_func = "y = cos(x)"
                base_expr = sp.cos(x)
                st.write(f"**원형 함수**: {base_func}")
                st.write(f"**함수**: f(x) = {format_expr(func_expr)}")
                
                st.write("- **특징**:")
                st.write("  - 정의역: 모든 실수")
                st.write("  - 치역: [-1, 1]")
                st.write("  - 주기: 2π")
            
            elif 'tan' in func_str:
                base_func = "y = tan(x)"
                base_expr = sp.tan(x)
                st.write(f"**원형 함수**: {base_func}")
                st.write(f"**함수**: f(x) = {format_expr(func_expr)}")
                
                st.write("- **특징**:")
                st.write("  - 정의역: x ≠ π/2 + nπ (n은 정수)")
                st.write("  - 치역: 모든 실수")
                st.write("  - 주기: π")
            
            # 3. 지수 함수 분석
            elif is_simple_exp(func_expr):
                base_func = "y = e^x"
                base_expr = sp.exp(x)
                st.write(f"**원형 함수**: {base_func}")
                st.write(f"**함수**: f(x) = {format_expr(func_expr)}")
                
                st.write("- **특징**:")
                st.write("  - 정의역: 모든 실수")
                st.write("  - 치역: (0, ∞)")
                st.write("  - 점근선: y = 0 (x축)")
                st.write("  - 증가/감소: 계수의 부호에 따라 결정")
            
            # 4. 로그 함수 분석
            elif is_simple_log(func_expr):
                base_func = "y = log(x)"
                base_expr = sp.log(x)
                st.write(f"**원형 함수**: {base_func}")
                st.write(f"**함수**: f(x) = {format_expr(func_expr)}")
                
                st.write("- **특징**:")
                st.write("  - 정의역: (0, ∞)")
                st.write("  - 치역: 모든 실수")
                st.write("  - 점근선: x = 0 (y축)")
            
            # 5. 초월 함수 (여러 함수의 조합)
            else:
                if is_transcendental_expr(func_expr):
                    st.write("**초월 함수입니다. 변환 분석에서는 자세한 원형 변환을 제공하지 않습니다.**")
                    st.write(f"**함수**: f(x) = {format_expr(func_expr)}")
                else:
                    st.write(f"**함수**: f(x) = {format_expr(func_expr)}")
                    st.write("- 변환 분석이 불가능합니다.")

            if base_expr is not None:
                if any(trig in func_str for trig in ['sin', 'cos', 'tan']):
                    x_vals_transform = np.linspace(-2*np.pi, 2*np.pi, 1200)
                else:
                    x_vals_transform = np.linspace(-5, 5, 1200)

                fig_orig, ax_orig = plt.subplots(figsize=(6, 4), constrained_layout=True)
                safe_plot_expr(ax_orig, base_expr, x_vals_transform, color='blue', label='원형 함수')
                ax_orig.set_title('원형 함수')
                ax_orig.set_ylabel('f(x)')

                fig_input, ax_input = plt.subplots(figsize=(6, 4), constrained_layout=True)
                safe_plot_expr(ax_input, func_expr, x_vals_transform, color='orange', label='입력 함수')
                ax_input.set_title('입력 함수')

                graph_col1, graph_col2 = st.columns([1, 1])
                with graph_col1:
                    st.pyplot(fig_orig)
                with graph_col2:
                    st.pyplot(fig_input)
                    st.write('**변형된 함수**')
                    st.write(f"f(x) = {format_expr(func_expr)}")

                info_col1, info_col2 = st.columns([3, 2])
                with info_col1:
                    st.write('**변환 정보**')
                    if transform_notes:
                        for note in transform_notes:
                            st.write(f"- {note}")
                    else:
                        st.write('- 평행이동 없음')
                    st.write(f"- 대칭 여부: {symmetry_note}")
                    st.write(f"- 원형 함수: {base_func}")
                with info_col2:
                    st.write('**그래프 설명**')
                    st.write('- 왼쪽은 원형 함수')
                    st.write('- 오른쪽은 입력 함수')

        # ============= 상세 정보 탭 =============
        with tab4:
            st.subheader("상세 정보")
            
            st.write("### 입력된 함수")
            st.code(f"f(x) = {func_expr}", language="python")
            
            st.write("### 정의역")
            st.write("정의역 분석은 자동으로 수행됩니다.")
            
            st.write("### 함수식 정리")
            try:
                simplified = simplify(func_expr)
                st.code(f"simplified: {simplified}", language="python")
            except:
                st.write("정리할 수 없습니다.")
            
            st.write("### 사용 가능한 함수 목록")
            st.markdown("""
            - **기본 연산**: `+`, `-`, `*`, `/`, `**` (거듭제곱)
            - **삼각함수**: `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`
            - **지수/로그**: `exp()`, `log()` (자연로그), `sqrt()`
            - **상수**: `pi`, `e` (자연상수)
            - **기타**: `abs()`, `Abs()`
            """)
    
    except ValueError as e:
        st.error(f"❌ 함수 입력 오류: {str(e)}")
        st.info("올바른 형식으로 다시 입력해주세요. 예: x**2, sin(x), exp(x-1)")
    
    except Exception as e:
        st.error(f"❌ 분석 중 오류 발생: {str(e)}")
        st.info("다른 함수를 시도해보세요.")

# 분석 예시
st.markdown("---")
st.write("### 🔎 분석 예시")
st.write("아래 버튼을 누르면 입력 상자에 예시 함수가 자동으로 채워지고 분석이 실행됩니다.")


# 고정 예시 함수 목록을 단순하게 사용합니다.
examples = {
    "1차 함수 예시": "2*x + 3",
    "2차 함수 예시": "x**2 - 4*x + 3",
    "3차 함수 예시": "x**3 - 3*x**2 + 2",
    "지수 함수 예시": "exp(x - 1) + 1",
    "로그 함수 예시": "log(x - 1) + 2",
    "sin 함수 예시": "sin(x - 1) + 1",
    "cos 함수 예시": "cos(x + 2) - 1",
    "tan 함수 예시": "tan(x - 1)",
    "유리 함수 예시": "1/(x - 2) + 1",
    "절댓값 예시": "abs(x + 1) - 2",
}

cols = st.columns(5)

def _select_example(e):
    st.session_state.example_selected = e
    st.session_state.func_input = e
    st.session_state.analyze = True

for idx, (name, expr) in enumerate(examples.items()):
    with cols[idx % 5]:
        st.button(name, key=f"example_{idx}", use_container_width=True, on_click=_select_example, args=(expr,))

if 'example_selected' in st.session_state:
    st.markdown(f"**선택된 분석 예시**: `{st.session_state.example_selected}`")
