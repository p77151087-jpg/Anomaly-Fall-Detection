# 正常資料異常偵測方法比較：事前規則

使用者授權範圍與根規格最新研究契約一致。第一階段run順序固定：SVM nu .01／gamma scale、.01／.001、.05／scale、.05／.001、.10／scale、.10／.001。只用原24 UR Train ADL與30幀isotropic 512D；核對Train-only Scaler與wide一致。RBF、tol=.001、shrinking=true、cache512MB、max_iter=-1，不擴大網格。分數為負decision_function，非機率。品質、原排程、strict >、K3不變。

新模型的Validation門檻枚舉所有合法影片K3分數及最低分數下一個float64。排序：包含unknown整体Accuracy、可判定Fall F1、全體Fall Recall、較少FP；門檻同分取較高值。候選平手最後按固定順序。基準仍用既有凍結門檻，不重新校準；這個新校準策略與歷史balanced Accuracy的差異明標，報告另列相同模型沿用歷史balanced校準的敏感度（僅診斷，不作追加候選或部署選擇）。

辨識採用：上述前四項嚴格優於最高wide才列研究採用；不增加unknown，不降低原品質。展示guard（六粗phase命中至少4、無粗正常前段早報、新救回片有當前良好骨架phase警示）另列、不當教師計分公式，未通過不自動換demo。時間改善可另列，不稱Accuracy提高。比較須包含全部23支、完整窗口排程與unknown，時間不是人工onset延遲。

SVM完成後，依据原Train／Validation的骨架、合法窗口、原七類頭獨立對照、特徵鄰近性及各模型分數鎖第二階段最多兩AE；架構／Loss／增強／參數在訓練前另存ae-lock。seed42、max100、patience10、min_delta1e-6、Adam lr.001、wd.001、batch32，正常Validation選checkpoint，不依Fall checkpoint。時間history不能跨影片／track／缺失或用未來；暖機保留為不可用端點，分母不縮。

第三階段只在前兩階段有支持表示瓶頸的診斷時另立一個候選。僅模型漏報／正常Loss下降／七類未判FALLING不足以判定表示問題；若同一表示的SVM或替代AE已救回漏報，優先視為計分方法的證據。若剩餘漏報主要同時有低品質／追蹤斷裂，且缺人工關節／時間GT與受控表示對照，不盲目微調。若能證明高品質連續骨架中的異常變化在特徵中被壓縮，再固定一個正常Train自監督任務與新權重／recipe／全新快取及異常模型。

停止：六SVM全部完成即停網格；兩AE完成即停容量、noise、history、seed調整。缺來源／hash錯誤／非有限值／時序因果失敗立即停止該階段，保留failed資料。每個run用exclusive新目錄；stage lock／結果不可覆寫。只讀核對可重跑。原Test全部保留未準備且無路徑，無新Test特徵讀取／評估。舊assets全hash快照，文件原版封存。正式Notebook／影片不更新、不公開。
