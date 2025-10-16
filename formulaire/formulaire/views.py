import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
from .utils import rag_pipeline  # ta fonction RAG


# --- Vue d'affichage du bot (mobile uniquement) ---
async def bot_views(request):
    user_agent = request.META.get('HTTP_USER_AGENT', '').lower()

    # Redirection vers Linktree pour les utilisateurs non mobiles
    if not any(keyword in user_agent for keyword in ["mobi", "android", "iphone"]):
        return redirect("https://linktr.ee/bureau_vision_nouvelle")

    return render(request, "bot_template.html")


# --- API de chat asynchrone ---
@csrf_exempt
async def chat_api(request):
    user_agent = request.META.get('HTTP_USER_AGENT', '').lower()

    # Redirection vers Linktree pour les utilisateurs non mobiles
    if not any(keyword in user_agent for keyword in ["mobi", "android", "iphone"]):
        return redirect("https://linktr.ee/bureau_vision_nouvelle")

    # On n'accepte que les requêtes POST
    if request.method != "POST":
        return JsonResponse({"error": "Méthode non autorisée"}, status=405)

    try:
        # ✅ Correction ici :
        # request.body est une propriété synchrone, PAS un awaitable.
        body = json.loads(request.body.decode("utf-8"))

        message = body.get("message", "")
        history = body.get("history", [])

        # ✅ rag_pipeline est async, donc on l'attend
        bot_reply = await rag_pipeline(message)

        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": bot_reply},
        ]

        return JsonResponse({
            "reply": bot_reply,
            "history": updated_history
        })

    except Exception as e:
        print("Erreur Vision Nouvelle Chat :", e)
        return JsonResponse({"error": str(e)}, status=500)
