import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .utils import rag_pipeline
from django.shortcuts import render, redirect



def bot_views(request):
    user_agent = request.META.get('HTTP_USER_AGENT', '').lower()
    if not any(keyword in user_agent for keyword in ["mobi", "android", "iphone"]):
        return redirect("https://linktr.ee/bureau_vision_nouvelle")
    
    return render(request, "bot_template.html")


# --- Configuration du modèle LLM Gemini ---
@csrf_exempt
def chat_api(request):
    user_agent = request.META.get('HTTP_USER_AGENT', '').lower()
    if not any(keyword in user_agent for keyword in ["mobi", "android", "iphone"]):
        return redirect("https://linktr.ee/bureau_vision_nouvelle")
    
    if request.method != "POST":
        return JsonResponse({"error": "Méthode non autorisée"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
        message = body.get("message", "")
        history = body.get("history", [])
        bot_reply = rag_pipeline(message)
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": bot_reply}
        ]
        return JsonResponse({
            "reply": bot_reply,
            "history": updated_history
        })
    except Exception as e:
        print("Erreur Vision Nouvelle Chat :", e)
        return JsonResponse({"error": str(e)}, status=500)
