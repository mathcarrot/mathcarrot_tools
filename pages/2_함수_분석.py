import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, sympify, sin, cos, tan, exp, log, diff, solve, limit, oo, simplify, Poly
import re

st.set_page_config(page_title="함수 분석 도구", layout="wide")
st.title("🔍 함수 분석 도구")

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

if 'example_selected' in st.session_state:
    st.session_state.func_input = st.session_state.example_selected

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
        
        # 탭 생성
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 그래프", "🔍 함수 분석", "📈 변환 분석", "🎯 특수점", "📋 상세 정보"]
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
                x_vals = np.linspace(x_min, x_max, 1000)
                y_vals = func_lambda(x_vals)
                y_vals = np.array(y_vals, dtype=np.complex128)
                
                # 실수값만 남기기
                real_mask = np.isfinite(y_vals.real) & np.isfinite(y_vals.imag) & (np.abs(y_vals.real) < 1000) & (np.abs(y_vals.imag) < 1e-8)
                x_plot = x_vals[real_mask]
                y_plot = y_vals.real[real_mask]
                
                if len(x_plot) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'f(x) = {func_input}')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='k', linewidth=0.8)
                    ax.axvline(x=0, color='k', linewidth=0.8)
                    ax.set_xlabel('x')
                    ax.set_ylabel('f(x)')
                    ax.set_title(f'함수의 그래프: f(x) = {func_input}')
                    
                    # 점근선 표시
                    for y_asym in horizontal_asymptotes:
                        ax.axhline(y=y_asym, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='수평 점근선' if y_asym == horizontal_asymptotes[0] else '')
                    
                    for x_asym in vertical_asymptotes:
                        if x_min <= x_asym <= x_max:
                            ax.axvline(x=x_asym, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='수직 점근선' if x_asym == vertical_asymptotes[0] else '')
                    
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
                    if y_auto:
                        # 극값과 변곡점을 포함한 y값 범위 계산
                        all_y_vals = list(y_plot)
                        
                        # 범위 내 극값 추가
                        all_y_vals.extend([p[1] for p in extrema if x_min <= p[0] <= x_max])
                        
                        # 범위 내 변곡점 추가
                        all_y_vals.extend([p[1] for p in inflections if x_min <= p[0] <= x_max])
                        
                        # y축 범위 계산
                        if len(all_y_vals) > 0:
                            y_min_data = np.min(all_y_vals)
                            y_max_data = np.max(all_y_vals)
                            y_range = y_max_data - y_min_data
                            
                            if y_range < 0.1:
                                y_range = 1.0
                            
                            # 여유 계산
                            margin = y_range * 0.15
                            
                            y_min_plot = y_min_data - margin
                            y_max_plot = y_max_data + margin
                            
                            ax.set_ylim(y_min_plot, y_max_plot)
                    else:
                        y_min = st.number_input("y축 최소값", value=float(np.min(y_plot)) if len(y_plot) > 0 else -10.0)
                        y_max = st.number_input("y축 최대값", value=float(np.max(y_plot)) if len(y_plot) > 0 else 10.0)
                        ax.set_ylim(y_min, y_max)
                    
                    st.pyplot(fig)
                else:
                    st.warning("주어진 범위에서 그려진 그래프가 없습니다.")
            except Exception as e:
                st.error(f"그래프 표시 중 오류: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        # ============= 함수 분석 탭 =============
        with tab2:
            st.subheader("함수 분석")
            
            # 원형 함수 인식
            st.write("### 원형 함수 (기본 함수) 인식")
            
            func_str = str(func_expr)
            func_type = "기타 함수"
            
            # 함수 유형 판정
            if func_expr.is_polynomial():
                poly = sp.Poly(func_expr, x)
                degree = poly.degree()
                func_type = f"{degree}차 다항 함수"
                st.write(f"**함수 유형**: {func_type}")
                st.write(f"**최고차항**: {poly.LC()}·x^{degree}")
                
            elif 'sin' in func_str or 'cos' in func_str or 'tan' in func_str:
                func_type = "삼각 함수"
                st.write(f"**함수 유형**: {func_type}")
                if 'sin' in func_str:
                    st.write("**기본 함수**: sin(x)")
                elif 'cos' in func_str:
                    st.write("**기본 함수**: cos(x)")
                else:
                    st.write("**기본 함수**: tan(x)")
                    
            elif 'exp' in func_str or func_expr.has(sp.exp):
                func_type = "지수 함수"
                st.write(f"**함수 유형**: {func_type}")
                st.write("**기본 함수**: e^x")
                
            elif 'log' in func_str or func_expr.has(sp.log):
                func_type = "로그 함수"
                st.write(f"**함수 유형**: {func_type}")
                st.write("**기본 함수**: ln(x) 또는 log(x)")
                
            else:
                st.write(f"**함수 유형**: {func_type}")
            
            # 함수의 성질
            st.write("### 함수의 기본 성질")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 우함수/기함수 판정
                func_f = func_expr.subs(x, x)
                func_f_neg = func_expr.subs(x, -x)
                
                if simplify(func_f - func_f_neg) == 0:
                    st.write("**대칭성**: 우함수 (짝함수) - y축 대칭")
                elif simplify(func_f + func_f_neg) == 0:
                    st.write("**대칭성**: 기함수 (홀함수) - 원점 대칭")
                else:
                    st.write("**대칭성**: 특별한 대칭성 없음")
            
            with col2:
                # 도함수 (1차)
                if f_prime is not None:
                    st.write(f"**도함수 f'(x)**: {f_prime}")
                else:
                    st.write("**도함수**: 계산 불가")
                
                # 2차 도함수
                if f_double_prime is not None:
                    st.write(f"**2차 도함수 f''(x)**: {f_double_prime}")
                else:
                    st.write("**2차 도함수**: 계산 불가")
            
            # 극값 정보 요약
            st.write("### 극값 정보")
            if extrema:
                max_points = [e for e in extrema if e[2] == 'max']
                min_points = [e for e in extrema if e[2] == 'min']
                
                if max_points:
                    st.write(f"**극대점 개수**: {len(max_points)}개")
                    for pt in max_points:
                        st.write(f"  - ({pt[0]:.4f}, {pt[1]:.4f})")
                
                if min_points:
                    st.write(f"**극소점 개수**: {len(min_points)}개")
                    for pt in min_points:
                        st.write(f"  - ({pt[0]:.4f}, {pt[1]:.4f})")
            else:
                st.write("극값이 없습니다.")
            
            # 변곡점 정보 요약
            st.write("### 변곡점 정보")
            if inflections:
                st.write(f"**변곡점 개수**: {len(inflections)}개")
                for inf in inflections:
                    st.write(f"  - ({inf[0]:.4f}, {inf[1]:.4f})")
            else:
                st.write("변곡점이 없습니다.")
        
        # ============= 변환 분석 탭 =============
        with tab3:
            st.subheader("함수의 변환 분석")
            st.write("### 원형에서의 변환")
            
            func_str = str(func_expr)
            base_func = None
            transformation_info = ""
            
            # 1. 다항 함수 분석
            if func_expr.is_polynomial():
                poly = Poly(func_expr, x)
                degree = poly.degree()
                base_func = f"y = x^{degree}"
                st.write(f"**원형 함수**: {base_func}")
                
                if degree == 1:
                    # 1차 함수: y = mx + b에서 y = x로의 변환
                    coeffs = poly.all_coeffs()
                    if len(coeffs) == 2:
                        m, b = coeffs
                        st.write(f"**함수**: f(x) = {m}x + {b}")
                        st.write(f"- 기울기: {m}")
                        if b > 0:
                            st.write(f"- y축으로 {b}만큼 위로 평행이동")
                        elif b < 0:
                            st.write(f"- y축으로 {abs(b)}만큼 아래로 평행이동")
                
                elif degree == 2:
                    # 2차 함수: 완전제곱식
                    coeffs = poly.all_coeffs()
                    if len(coeffs) == 3:
                        a_coef, b_coef, c_coef = coeffs
                        h = -b_coef / (2*a_coef)
                        k = func_expr.subs(x, h)
                        
                        st.write(f"**표준형**: f(x) = {a_coef}(x - ({h}))² + ({k})")
                        st.write(f"- **평행이동**:")
                        if h > 0:
                            st.write(f"  - x축으로 {float(h):.4f}만큼 오른쪽")
                        elif h < 0:
                            st.write(f"  - x축으로 {float(abs(h)):.4f}만큼 왼쪽")
                        if k > 0:
                            st.write(f"  - y축으로 {float(k):.4f}만큼 위")
                        elif k < 0:
                            st.write(f"  - y축으로 {float(abs(k)):.4f}만큼 아래")
                        
                        if a_coef < 0:
                            st.write(f"- **대칭이동**: x축에 대해 대칭반사")
                
                elif degree == 3:
                    # 3차 함수
                    st.write(f"**함수**: f(x) = {func_expr}")
                    # 간단한 형태로 표시
                    coeffs = poly.all_coeffs()
                    if len(coeffs) >= 2:
                        st.write(f"- 최고차 계수: {coeffs[0]}")
                        if coeffs[0] > 0:
                            st.write(f"- x → ∞일 때 f(x) → ∞, x → -∞일 때 f(x) → -∞")
                        else:
                            st.write(f"- x → ∞일 때 f(x) → -∞, x → -∞일 때 f(x) → ∞")
                
                else:
                    # n차 함수
                    st.write(f"**함수**: f(x) = {func_expr}")
                    coeffs = poly.all_coeffs()
                    st.write(f"- 차수: {degree}")
                    st.write(f"- 최고차 계수: {coeffs[0]}")
            
            # 2. 삼각 함수 분석
            elif 'sin' in func_str:
                base_func = "y = sin(x)"
                st.write(f"**원형 함수**: {base_func}")
                st.write(f"**함수**: f(x) = {func_expr}")
                
                # 진폭, 주기, 평행이동 분석
                st.write("- **특징**:")
                st.write("  - 정의역: 모든 실수")
                st.write("  - 치역: [-1, 1]")
                st.write("  - 주기: 2π")
            
            elif 'cos' in func_str:
                base_func = "y = cos(x)"
                st.write(f"**원형 함수**: {base_func}")
                st.write(f"**함수**: f(x) = {func_expr}")
                
                st.write("- **특징**:")
                st.write("  - 정의역: 모든 실수")
                st.write("  - 치역: [-1, 1]")
                st.write("  - 주기: 2π")
            
            elif 'tan' in func_str:
                base_func = "y = tan(x)"
                st.write(f"**원형 함수**: {base_func}")
                st.write(f"**함수**: f(x) = {func_expr}")
                
                st.write("- **특징**:")
                st.write("  - 정의역: x ≠ π/2 + nπ (n은 정수)")
                st.write("  - 치역: 모든 실수")
                st.write("  - 주기: π")
            
            # 3. 지수 함수 분석
            elif 'exp' in func_str or func_expr.has(sp.exp):
                base_func = "y = e^x"
                st.write(f"**원형 함수**: {base_func}")
                st.write(f"**함수**: f(x) = {func_expr}")
                
                st.write("- **특징**:")
                st.write("  - 정의역: 모든 실수")
                st.write("  - 치역: (0, ∞)")
                st.write("  - 점근선: y = 0 (x축)")
                st.write("  - 증가/감소: 계수의 부호에 따라 결정")
            
            # 4. 로그 함수 분석
            elif 'log' in func_str or func_expr.has(sp.log):
                base_func = "y = log(x)"
                st.write(f"**원형 함수**: {base_func}")
                st.write(f"**함수**: f(x) = {func_expr}")
                
                st.write("- **특징**:")
                st.write("  - 정의역: (0, ∞)")
                st.write("  - 치역: 모든 실수")
                st.write("  - 점근선: x = 0 (y축)")
            
            # 5. 초월 함수 (여러 함수의 조합)
            else:
                if any(trig in func_str for trig in ['sin', 'cos', 'tan', 'exp', 'log']):
                    st.write("**원형 함수**: 없음 (초월함수)")
                    st.write(f"**함수**: f(x) = {func_expr}")
                    st.write("- **설명**: 이 함수는 여러 초월함수가 결합된 형태입니다.")
                    st.write("- 기본 함수들의 조합으로 이루어진 복잡한 함수입니다.")
                else:
                    st.write(f"**함수**: f(x) = {func_expr}")
                    st.write("- 변환 분석이 불가능합니다.")
        
        # ============= 특수점 탭 =============
        with tab4:
            st.subheader("특수점 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### x절편 (근/영점)")
                try:
                    zeros = solve(func_expr, x)
                    if zeros:
                        has_zero = False
                        for i, zero in enumerate(zeros):
                            try:
                                zero_val = complex(zero)
                                if abs(zero_val.imag) < 1e-10:  # 실근만 표시
                                    st.write(f"x = {zero_val.real:.6f}")
                                    has_zero = True
                            except:
                                try:
                                    st.write(f"x = {float(zero.evalf()):.6f}")
                                    has_zero = True
                                except:
                                    st.write(f"x = {zero}")
                                    has_zero = True
                        if not has_zero:
                            st.write("실수 근이 없습니다.")
                    else:
                        st.write("실수 근이 없습니다.")
                except Exception as e:
                    st.write("근을 계산할 수 없습니다.")
            
            with col2:
                st.write("### y절편")
                try:
                    y_intercept = func_expr.subs(x, 0)
                    if hasattr(y_intercept, 'evalf'):
                        y_val = float(y_intercept.evalf())
                    else:
                        y_val = float(y_intercept)
                    st.write(f"f(0) = {y_val:.6f}")
                except:
                    st.write("y절편을 계산할 수 없습니다.")
            
            # 극값 (극대, 극소)
            st.write("### 극값")
            if extrema:
                has_extrema_display = False
                for ex in extrema:
                    if ex[2] == 'max':
                        st.write(f"**극대점**: x = {ex[0]:.6f}, f(x) = {ex[1]:.6f}")
                        has_extrema_display = True
                    elif ex[2] == 'min':
                        st.write(f"**극소점**: x = {ex[0]:.6f}, f(x) = {ex[1]:.6f}")
                        has_extrema_display = True
                
                if not has_extrema_display:
                    st.write("극값이 없습니다.")
            else:
                st.write("극값이 없습니다.")
            
            # 변곡점
            st.write("### 변곡점")
            if inflections:
                for inf in inflections:
                    st.write(f"x = {inf[0]:.6f}, f(x) = {inf[1]:.6f}")
            else:
                st.write("변곡점이 없습니다.")
            
            # 점근선
            st.write("### 점근선")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**수평 점근선 (y = ?)**")
                if horizontal_asymptotes:
                    for h_asym in horizontal_asymptotes:
                        st.write(f"y = {h_asym:.6f}")
                else:
                    st.write("없음")
            
            with col2:
                st.write("**수직 점근선 (x = ?)**")
                if vertical_asymptotes:
                    for v_asym in vertical_asymptotes:
                        st.write(f"x = {v_asym:.6f}")
                else:
                    st.write("없음")
        
        # ============= 상세 정보 탭 =============
        with tab5:
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
for idx, (name, expr) in enumerate(examples.items()):
    with cols[idx % 5]:
        if st.button(name, key=f"example_{idx}", use_container_width=True):
            st.session_state.example_selected = expr
            st.session_state.analyze = True

if 'example_selected' in st.session_state:
    st.markdown(f"**선택된 분석 예시**: `{st.session_state.example_selected}`")
