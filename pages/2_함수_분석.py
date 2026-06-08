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

with col1:
    func_input = st.text_input(
        "함수를 입력하세요",
        value="x**2 - 2*x + 1",
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
        
        # 함수를 sympy 식으로 변환
        func_expr = sympify(func_input)
        
        st.success("✅ 함수가 올바르게 입력되었습니다!")
        
        # 탭 생성
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 그래프", "🔍 함수 분석", "📈 변환 분석", "🎯 특수점", "📋 상세 정보"]
        )
        
        # ============= 그래프 탭 =============
        with tab1:
            st.subheader("함수의 그래프")
            
            # x의 범위 설정
            col1, col2, col3 = st.columns(3)
            with col1:
                x_min = st.number_input("x의 최소값", value=-5.0)
            with col2:
                x_max = st.number_input("x의 최대값", value=5.0)
            with col3:
                y_auto = st.checkbox("y축 자동 조정", value=True)
            
            try:
                # 함수를 그릴 수 있도록 변환
                func_lambda = sp.lambdify(x, func_expr, 'numpy')
                
                # x 데이터 생성
                x_vals = np.linspace(x_min, x_max, 1000)
                y_vals = func_lambda(x_vals)
                
                # 이상값 처리 (너무 큰 값, NaN 등)
                valid_mask = np.isfinite(y_vals) & (np.abs(y_vals) < 1000)
                x_plot = x_vals[valid_mask]
                y_plot = y_vals[valid_mask]
                
                if len(x_plot) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'f(x) = {func_input}')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='k', linewidth=0.8)
                    ax.axvline(x=0, color='k', linewidth=0.8)
                    ax.set_xlabel('x')
                    ax.set_ylabel('f(x)')
                    ax.set_title(f'함수의 그래프: f(x) = {func_input}')
                    ax.legend()
                    
                    if not y_auto:
                        y_min = st.number_input("y의 최소값", value=float(np.min(y_plot)))
                        y_max = st.number_input("y의 최대값", value=float(np.max(y_plot)))
                        ax.set_ylim(y_min, y_max)
                    
                    st.pyplot(fig)
                else:
                    st.warning("주어진 범위에서 그려진 그래프가 없습니다.")
            except Exception as e:
                st.error(f"그래프 표시 중 오류: {str(e)}")
        
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
                try:
                    f_prime = diff(func_expr, x)
                    st.write(f"**도함수 f'(x)**: {f_prime}")
                except:
                    st.write("**도함수**: 계산 불가")
        
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
                        for i, zero in enumerate(zeros):
                            try:
                                zero_val = complex(zero)
                                if abs(zero_val.imag) < 1e-10:  # 실근만 표시
                                    st.write(f"x = {zero_val.real:.6f}")
                            except:
                                st.write(f"x = {zero}")
                    else:
                        st.write("실수 근이 없습니다.")
                except:
                    st.write("근을 계산할 수 없습니다.")
            
            with col2:
                st.write("### y절편")
                try:
                    y_intercept = func_expr.subs(x, 0)
                    st.write(f"f(0) = {y_intercept}")
                except:
                    st.write("y절편을 계산할 수 없습니다.")
            
            # 극값 (극대, 극소)
            st.write("### 극값")
            try:
                f_prime = diff(func_expr, x)
                critical_points = solve(f_prime, x)
                
                if critical_points:
                    # 2차 도함수로 극값 판정
                    f_double_prime = diff(f_prime, x)
                    
                    has_extrema = False
                    for crit_point in critical_points:
                        try:
                            crit_val = float(crit_point)
                            second_deriv_val = float(f_double_prime.subs(x, crit_point))
                            func_val = float(func_expr.subs(x, crit_point))
                            
                            if second_deriv_val > 0:
                                st.write(f"**극소점**: x = {crit_val:.6f}, f(x) = {func_val:.6f}")
                                has_extrema = True
                            elif second_deriv_val < 0:
                                st.write(f"**극대점**: x = {crit_val:.6f}, f(x) = {func_val:.6f}")
                                has_extrema = True
                        except:
                            pass
                    
                    if not has_extrema:
                        st.write("극값이 없습니다.")
                else:
                    st.write("극값이 없습니다.")
            except:
                st.write("극값을 계산할 수 없습니다.")
            
            # 변곡점
            st.write("### 변곡점")
            try:
                f_prime = diff(func_expr, x)
                f_double_prime = diff(f_prime, x)
                inflection_points = solve(f_double_prime, x)
                
                if inflection_points:
                    has_inflection = False
                    for inf_point in inflection_points:
                        try:
                            inf_val = float(inf_point)
                            func_val = float(func_expr.subs(x, inf_point))
                            st.write(f"x = {inf_val:.6f}, f(x) = {func_val:.6f}")
                            has_inflection = True
                        except:
                            pass
                    
                    if not has_inflection:
                        st.write("변곡점이 없습니다.")
                else:
                    st.write("변곡점이 없습니다.")
            except:
                st.write("변곡점을 계산할 수 없습니다.")
            
            # 점근선
            st.write("### 점근선")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**수평 점근선**")
                try:
                    limit_pos_inf = limit(func_expr, x, oo)
                    limit_neg_inf = limit(func_expr, x, -oo)
                    
                    has_horizontal = False
                    
                    if limit_pos_inf.is_finite:
                        st.write(f"x → +∞: y = {limit_pos_inf}")
                        has_horizontal = True
                    
                    if limit_neg_inf.is_finite and limit_neg_inf != limit_pos_inf:
                        st.write(f"x → -∞: y = {limit_neg_inf}")
                        has_horizontal = True
                    
                    if not has_horizontal:
                        st.write("없음")
                except:
                    st.write("계산할 수 없습니다.")
            
            with col2:
                st.write("**수직 점근선**")
                try:
                    # 분모가 0이 되는 점 찾기 (유리함수의 경우)
                    vertical_asymptotes = []
                    
                    # 간단한 유리함수 분석
                    if '/' in func_input:
                        # 분모 찾기
                        numer, denom = sp.fraction(func_expr)
                        denom_zeros = solve(denom, x)
                        
                        has_vertical = False
                        for zero in denom_zeros:
                            try:
                                zero_val = complex(zero)
                                if abs(zero_val.imag) < 1e-10:
                                    st.write(f"x = {zero_val.real:.6f}")
                                    has_vertical = True
                            except:
                                st.write(f"x = {zero}")
                                has_vertical = True
                        
                        if not has_vertical:
                            st.write("없음")
                    else:
                        st.write("없음")
                except:
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

# 예제
st.markdown("---")
st.write("### 📝 사용 예제")

examples = {
    "1차 함수": "2*x + 3",
    "2차 함수": "x**2 - 4*x + 3",
    "3차 함수": "x**3 - 2*x",
    "절댓값": "abs(x)",
    "유리함수": "1/x",
    "유리함수 2": "1/(x-2)",
    "삼각함수": "sin(x)",
    "지수함수": "exp(x)",
    "로그함수": "log(x)",
    "복합함수": "x*exp(-x)",
}

cols = st.columns(5)
for idx, (name, expr) in enumerate(examples.items()):
    with cols[idx % 5]:
        if st.button(name, key=f"example_{idx}", use_container_width=True):
            st.session_state.example_selected = expr
            st.rerun()

if 'example_selected' in st.session_state:
    st.markdown(f"**선택된 예제**: `{st.session_state.example_selected}`")
