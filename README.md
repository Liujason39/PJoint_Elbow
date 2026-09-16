# planar-linkage 0.1.0

伸縮致動器驅動平面四連桿：位置、速度、加速度、Jacobian 與準靜態力傳導。
Python >= 3.10；唯一執行期相依套件為 NumPy。

## 模型與座標

固定點 O2=(0,0)、O4=(d,0)、O1=(u,v)。
定長桿 O2A=a、AB=b、O4B=c；輸入 s=|O1A|。
O1A 是兩端可轉動的伸縮致動器，不是方向固定的滑軌。
A 為致動器、O2A、AB 共用的轉動接點；B 為 AB、O4B 的轉動接點。
所有桿為剛體，機構為平面模型。交叉處不新增接點。

angles、omega、alpha 的順序固定為 theta2、theta3、theta4，
分別是 O2→A、A→B、O4→B 相對 +x 的逆時針角度。
角度用 rad。建議長度 m、時間 s、力 N、扭矩 N m；也可採其他一致單位。
s 是兩鉸點之間的完整長度，不是相對行程；若輸入行程 x，請用 s=s0+x。

## 安裝與驗證

解壓縮後，在此 README 所在目錄執行：

```bash
python -m pip install .
python -m unittest discover -s tests -v
python examples/demo.py
```

離線且已裝 NumPy / setuptools / wheel 時，可用：

```bash
python -m pip install --no-build-isolation --no-deps .
```

## 使用

```python
from planar_linkage import Mechanism

m = Mechanism(a=1.0, b=2.0, c=1.8, d=2.2, u=-0.8, v=0.3)
s = 1.2
branches = (1, 1)
state, omega, alpha = m.motion(s, s_dot=0.02, s_ddot=0.005,
                                branches=branches)
print(state.A, state.B, state.angles)
print(omega, alpha)
J = m.jacobian(s, branches)       # shape (3,), d angles / ds
JB = m.point_jacobian(s, 'B', branches)  # shape (2,), d B / ds
p, velocity, acceleration = m.point_motion(s, 0.02, 0.005, 'B', branches)
f = m.input_force(s, force_B=(0, -100), branches=branches)
```

以上尺寸僅為可運動的示例，請替換成實際尺寸。

## 組裝分支與連續運動

`branches=(branch_A, branch_B)` 各為 +1 或 -1：

- branch_A：A 位於有向直線 O2→O1 的左側 (+1) 或右側 (-1)。
- branch_B：B 位於有向直線 A→O4 的左側 (+1) 或右側 (-1)。

這些符號不等於「一般/交叉」構形標籤。有四種候選組合，但不一定全部可達。
請用 position 回傳座標確認實際組裝。一般區間保持分支；穿過分支重合點時
可能需改分支，但本套件不會自動決定死點之後的物理路徑。
`position` 允許相切位置；`jacobian`、`motion` 與 `input_force` 在奇異/近奇異位置
拋出 `SingularityError`，不返回假性的有限解。
不同圓心重合的退化情況拋出 `GeometryError`。

多個輸入位置可逐點呼叫；需要連續角度畫圖時：

```python
import numpy as np
samples = [m.position(s, branches) for s in np.linspace(0.8, 1.5, 101)]
angles = np.unwrap(np.array([p.angles for p in samples]), axis=0)
```

不可達位置會拋出 `GeometryError`，掃描行程時應由使用者捕捉並處理。
套件不模擬連桿碰撞、接頭行程限制或自動通過死點。

## 數學方法

令 q=(theta2,theta3,theta4)，約束 F(q,s)=0：

```
F1 = |a(cos theta2, sin theta2) - (u,v)|^2 - s^2
F2 = a cos theta2 + b cos theta3 - c cos theta4 - d
F3 = a sin theta2 + b sin theta3 - c sin theta4
```

位置由兩次兩圓交點求得，不用非線性迭代。
C = dF/dq，解 C J = (2s,0,0) 得 J=dq/ds。
速度 q_dot=J s_dot。二次微分完整約束得到角加速度，包含平方速度項。
求解前用幾何尺度將矩陣各列無因次化，再以奇異值比偵測近奇異位置。
預設門檻 singularity_tolerance=1e-10 可調；門檻不是機構安全界限。

接點 Jacobian：JA=a(-sin theta2,cos theta2)J[0]，
JB=c(-sin theta4,cos theta4)J[2]。

## 力傳導與符號

`input_force` 回傳使 s 增加為正的致動器推力。
`torques=(tau2,tau3,tau4)` 是施加在各桿上的外部逆時針扭矩；
`force_A`、`force_B` 是施加在機構上的外部 Cartesian 力。
若多種負載同時傳入，會相加，避免重複輸入同一負載的等效力與等效扭矩。

虛功平衡：

```
f = -(J dot torques + JA dot force_A + JB dot force_B)
```

若僅指定機構對負載的輸出扭矩 T_out，使用
`input_force(s, torques=(0,0,-T_out))`。
僅有此負載且 J[2] 非零時，T_out=f/J[2]。
這是理想準靜態結果，不含自動計算的重力、摩擦、質量或慣量，
也不提供桿內力、鉸鏈反力、結構強度或動力學求解。

## API

| 方法 | 回傳 |
|---|---|
| position(s, branches) | State：s、angles、A、B、branches |
| residual(angles,s) | 三個閉迴路殘差，第一項長度平方、其餘長度 |
| jacobian(s,branches) | 3 個角度對 s 的導數 |
| motion(s,s_dot,s_ddot,branches) | State、角速度、角加速度 |
| point_jacobian(s,point,branches) | A 或 B 的位置對 s 導數 |
| point_motion(s,s_dot,s_ddot,point,branches) | 位置、速度、加速度 |
| input_force(s,*,torques,force_A,force_B,branches) | 平衡負載所需推力 |

## 驗證內容

使用標準庫 unittest，測試 4 個組裝分支 × 10 個輸入位置的閉合殘差、
Jacobian 中央差分與位置二階時間差分；另測試外力/扭矩的虛功率平衡、
單位縮放、不可達輸入、三角形死點與四連桿死點。
數值差分使用角度環繞處理，避免 atan2 的 ±pi 跳變。
